use std::collections::HashMap;

use ferrotorch::{
    distributions::{Categorical, Distribution},
    from_vec,
    hub::{HubCache, hf_download_model},
    nn::{Buffer, Embedding, ModuleList, StateDict, clip_grad_norm_, init},
    optim::AdamWConfig,
    prelude::*,
    serialize::load_safetensors,
    tokenize::{Tokenizer, decode, encode, load_tokenizer},
};

use std::fs;

fn main() -> FerrotorchResult<()> {
    let in_str = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Hello, I'm a language model,".to_string());

    let res = do_gpt2(&in_str);
    println!("{res:?}");
    Ok(())
}

fn do_gpt2(_in_str: &str) -> FerrotorchResult<()> {
    let cache = HubCache::with_default_dir();
    let dir = hf_download_model("openai-community/gpt2", "main", &cache)?;

    println!("Getting weights");
    let _state_dict = load_safetensors::<f32>(&dir.join("model.safetensors"))?;
    let tok = load_tokenizer(&dir.join("tokenizer.json"))?;

    let mut gpt: GPT<f32> = GPT::new(GPTConfig::GPT2);

    // println!("Loading weights into gpt");
    // let gpt = gpt.load_from_statedict(&state_dict)?;

    let device = if cuda_available() && ferrotorch::gpu::init_cuda_backend().is_ok() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };

    let b = 4;
    let t = 32;

    // TODO should probably pre-tokenize
    let (x, y) = get_batch::<f32>(b, t, &tok, device)?;

    let gpt = gpt.set_tok(tok)?;
    let gpt = gpt.fresh_params()?;

    gpt.move_to_device(device)?;

    // let size_batch = 1;
    // let decode_n_toks = 5;

    // let out = no_grad(|| gpt.pipeline(in_str, size_batch, decode_n_toks))?;

    // for s in out {
    //     println!("> {}", s);
    // }
    // println!("logits: {:?}", logits.data_vec()?);

    let mut optimizer = AdamW::new(
        gpt.parameters().into_iter().cloned().collect(),
        AdamWConfig::default().with_lr(3e-4).with_betas((0.9, 0.95)),
    );

    // Do the overfit
    for i in 0..50 {
        optimizer.zero_grad();

        let (_logits, loss) = gpt.forward_with_loss(&x, &y)?;

        println!("step: {i}, loss: {:?}", loss.data_vec()?);

        backward(&loss)?;
        let _total_norm = clip_grad_norm_(&gpt.parameters(), 1.0, 2.0)?;
        optimizer.step();
    }

    Ok(())
}

fn get_batch<T: Float>(
    b: usize,
    t: usize,
    tok: &Tokenizer,
    device: Device,
) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
    let text = fs::read_to_string("./data/input.txt").unwrap();
    let ids: Vec<u32> = encode(&tok, text.as_str(), false)?;
    let mut x = Vec::with_capacity(b * t);
    let mut y = Vec::with_capacity(b * t);
    let mut i = 0;
    for _ in 0..b {
        for k in 0..t {
            x.push(T::from(ids[i + k]).unwrap());
            y.push(T::from(ids[i + k + 1]).unwrap());
        }
        i += t;
        if i + t + 1 >= ids.len() {
            // TODO check for off-by-one
            break;
        }
    }

    let x = from_vec(x, &[b, t])?;
    let y = from_vec(y, &[b, t])?;

    Ok((x.to(device)?, y.to(device)?))
}

#[derive(Copy, Clone)]
struct GPTConfig {
    block_size: usize,
    vocab_size: usize,
    n_layer: usize,
    n_head: usize,
    n_embd: usize,
}

impl GPTConfig {
    const GPT2: Self = Self {
        block_size: 1024,
        vocab_size: 50257,
        n_layer: 12,
        n_head: 12,
        n_embd: 768,
    };
}

#[derive(Module)]
struct GPT<T: Float> {
    config: GPTConfig,
    #[submodule]
    transformer: Transformer<T>,
    #[submodule]
    lm_head: Linear<T>,
    tok: Option<Tokenizer>,
    cur_device: Option<Device>, // If moved
    training: bool,
}

impl<T: Float> GPT<T> {
    fn new(config: GPTConfig) -> Self {
        Self {
            config,
            transformer: Transformer::new(config),
            lm_head: Linear::new(config.n_embd, config.vocab_size, false).unwrap(),
            tok: None,
            cur_device: None,
            training: false,
        }
    }

    fn transform_sd(state_dict: &StateDict<T>) -> FerrotorchResult<StateDict<T>> {
        // "ends with" items that need to be transposed from the HF weights
        let transposed_elems = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ];

        // "ends with" items to ignore
        let to_ignore = [".attn.bias"];

        let mut new_state_dict = StateDict::new();

        for (k, v) in state_dict {
            let needs_ignore = to_ignore.iter().any(|elem| k.ends_with(elem));
            if needs_ignore {
                continue;
            }

            let needs_transpose = transposed_elems.iter().any(|elem| k.ends_with(elem));
            let tensor = if needs_transpose {
                &v.transpose(0, 1)?.contiguous()?
            } else {
                v
            };

            new_state_dict.insert(k.to_string(), tensor.clone());
        }

        Ok(new_state_dict)
    }

    fn load_from_statedict(&mut self, state_dict: &StateDict<T>) -> FerrotorchResult<&mut Self> {
        // Do transpose and drop
        let state_dict = GPT::transform_sd(state_dict)?;

        load_submodule_fields(&mut self.transformer.h, &state_dict, "h")?;

        load_submodule_fields(&mut self.transformer.wpe, &state_dict, "wpe")?;
        load_submodule_fields(&mut self.transformer.wte, &state_dict, "wte")?;
        load_submodule_fields(&mut self.transformer.ln_f, &state_dict, "ln_f")?;

        self.lm_head.weight = self.transformer.wte.weight.clone();

        Ok(self)
    }

    fn fresh_params(&mut self) -> FerrotorchResult<&mut Self> {
        // Use normal distributiopn, sample with mean 0, std 0.02
        init::normal(&mut self.transformer.wte.weight, 0.0, 0.02)?;
        init::normal(&mut self.transformer.wpe.weight, 0.0, 0.02)?;

        let blocks = (0..self.config.n_layer)
            .map(|_| Box::new(Block::new(self.config)) as Box<dyn Module<T>>)
            .collect::<Vec<_>>();
        self.transformer.h = ModuleList::new(blocks);

        // TODO we will want to keep this tethered?
        // Probably needs retethering after move
        // And special treatment during train?
        self.lm_head.weight = self.transformer.wte.weight.clone();

        Ok(self)
    }

    fn set_tok(&mut self, tok: Tokenizer) -> FerrotorchResult<&mut Self> {
        self.tok = Some(tok);
        Ok(self)
    }

    fn forward(&self, idx: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let [b, t] = idx.shape() else { panic!("TODO") };

        let (b, t) = (*b, *t);

        let idx_flat = idx.reshape_t(&[(b * t) as isize])?;
        let tok_emb = self.transformer.wte.forward(&idx_flat)?.reshape_t(&[
            b as isize,
            t as isize,
            self.config.n_embd as isize,
        ])?;

        let pos = arange(
            <T as ferrotorch::Element>::zero(),
            T::from(t).unwrap(),
            <T as ferrotorch::Element>::one(),
        )?
        .to(idx.device())?;
        let pos_emb = self.transformer.wpe.forward(&pos)?;

        let mut x = tok_emb.add_t(&pos_emb)?;

        for i in 0..self.transformer.h.len() {
            let block = self.transformer.h.get(i).unwrap();
            x = block.forward(&x)?;
        }

        x = self.transformer.ln_f.forward(&x)?;
        let logits = self.lm_head.forward(&x)?;
        Ok(logits)
    }

    fn forward_with_loss(
        &self,
        idx: &Tensor<T>,
        targets: &Tensor<T>,
    ) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
        let logits = self.forward(idx)?;
        let [b, t, v] = logits.shape() else { panic!() };
        let (b, t, v) = (*b, *t, *v);

        let logits_2d = logits.reshape_t(&[(b * t) as isize, v as isize])?;
        let targets_1d = targets.reshape_t(&[(b * t) as isize])?;
        let loss = CrossEntropyLoss::default().forward(&logits_2d, &targets_1d)?;
        Ok((logits, loss))
    }

    fn pipeline(
        &self,
        in_str: &str,
        size_batch: usize,
        decode_n_toks: usize,
    ) -> FerrotorchResult<Vec<String>> {
        let ids: Vec<u32> = encode(&self.tok.as_ref().unwrap(), in_str, false)?;
        let t = ids.len();

        let data: Vec<T> = ids
            .repeat(size_batch)
            .into_iter()
            .map(|id| T::from(id).unwrap())
            .collect();
        let batch: Tensor<T> = from_vec(data, &[size_batch, t])?;

        // Before forward passes, make sure batch is on device
        // Cheap if it is already, so its okay to do unnecessarily
        let mut batch = if let Some(dev) = self.cur_device {
            batch.to(dev)?
        } else {
            batch
        };

        while batch.size()[1] < (t + decode_n_toks) {
            let logits = &self.forward(&batch)?;
            let cur_t = batch.size()[1];
            let logits = logits.narrow(1, cur_t - 1, 1)?.contiguous()?.squeeze_t(1)?;
            let probs = logits.softmax()?;
            let (topk_probs, topk_indices) = topk(&probs.cpu()?, 50, true)?;

            let b = topk_probs.shape()[0];
            let mut next_ids = Vec::<i64>::with_capacity(b);
            for row in 0..b {
                // Sample can't be seeded in current ferrotorch
                // And also, is a different algo than pytorch anyways
                let row_probs = topk_probs.narrow(0, row, 1)?.contiguous()?.squeeze_t(0)?;
                let pick = Categorical::new(row_probs)?.sample(&[1])?;

                let local_idx = pick.data_vec()?[0].to_usize().unwrap();
                let token_id = topk_indices[row * 50 + local_idx] as i64;
                next_ids.push(token_id);
            }

            let new_col_data: Vec<T> = next_ids.iter().map(|&id| T::from(id).unwrap()).collect();
            let new_col = from_vec(new_col_data, &[b, 1])?.to(batch.device())?;

            batch = cat(&[batch, new_col], 1)?;
        }

        let (b, t) = (batch.shape()[0], batch.shape()[1]);
        let mut out = Vec::<String>::new();

        let flat = batch.data_vec()?;
        for row in 0..b {
            let ids: Vec<u32> = flat[row * t..(row + 1) * t]
                .iter()
                .map(|&x| x.to_u32().unwrap())
                .collect();
            let text = decode(self.tok.as_ref().unwrap(), &ids, true)?;
            out.push(text);
        }

        Ok(out)
    }

    // Probably prone to half-failed state?
    fn move_to_device(&mut self, device: Device) -> FerrotorchResult<()> {
        self.lm_head.to_device(device)?;
        self.transformer.wpe.to_device(device)?;
        self.transformer.wte.to_device(device)?;
        self.transformer.ln_f.to_device(device)?;
        for i in 0..self.transformer.h.len() {
            self.transformer.h.get_mut(i).unwrap().to_device(device)?;
        }

        self.cur_device = Some(device);

        Ok(())
    }
}

fn load_submodule_fields<T: Float, M: Module<T>>(
    module: &mut M,
    state_dict: &StateDict<T>,
    parent_key: &str,
) -> FerrotorchResult<()> {
    let stripped_map: HashMap<String, Tensor<T>> = state_dict
        .iter()
        .filter_map(|(k, v)| {
            k.strip_prefix(&format!("{parent_key}."))
                .map(|remaining| (remaining.to_string(), v.clone()))
        })
        .collect();
    module.load_state_dict(&stripped_map, false)
}

#[derive(Module)]
struct Transformer<T: Float> {
    #[submodule]
    wte: Embedding<T>,
    #[submodule]
    wpe: Embedding<T>,
    #[submodule]
    h: ModuleList<T>,
    #[submodule]
    ln_f: LayerNorm<T>,
    training: bool,
}

impl<T: Float> Transformer<T> {
    fn new(config: GPTConfig) -> Self {
        let blocks = (0..config.n_layer)
            .map(|_| Box::new(Block::new(config)) as Box<dyn Module<T>>)
            .collect::<Vec<_>>();

        Self {
            wte: Embedding::new(config.vocab_size, config.n_embd, None).unwrap(),
            wpe: Embedding::new(config.block_size, config.n_embd, None).unwrap(),
            h: ModuleList::new(blocks),
            ln_f: LayerNorm::new(vec![config.n_embd], 1e-5, true).unwrap(),
            training: false,
        }
    }
}

struct Block<T: Float> {
    ln_1: LayerNorm<T>,
    attn: CausalSelfAttention<T>,
    ln_2: LayerNorm<T>,
    mlp: MLP<T>,
}

impl<T: Float> Block<T> {
    fn new(config: GPTConfig) -> Self {
        let mut block = Self {
            ln_1: LayerNorm::new(vec![config.n_embd], 1e-5, true).unwrap(), // TODO check learnable?
            attn: CausalSelfAttention::new(config),
            ln_2: LayerNorm::new(vec![config.n_embd], 1e-5, true).unwrap(), // TODO check learnable?
            mlp: MLP::new(config),
        };

        let proj_std = 0.02 * (2.0 * config.n_layer as f64).powf(-0.5);
        init::normal(&mut block.attn.c_attn.weight, 0.0, 0.02).unwrap();
        init::normal(&mut block.attn.c_proj.weight, 0.0, proj_std).unwrap();
        init::normal(&mut block.mlp.c_fc.weight, 0.0, 0.02).unwrap();
        init::normal(&mut block.mlp.c_proj.weight, 0.0, proj_std).unwrap();

        block
    }
}

impl<T: Float> Module<T> for Block<T> {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let x: Tensor<T> = {
            let normalized = self.ln_1.forward(&x)?;
            let attn_out = self.attn.forward(&normalized)?;
            x.add_t(&attn_out)?
        };

        let x: Tensor<T> = {
            let normalized = self.ln_2.forward(&x)?;
            let mlp_out = self.mlp.forward(&normalized)?;
            x.add_t(&mlp_out)?
        };

        Ok(x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.ln_1.parameters());
        out.extend(self.attn.parameters());
        out.extend(self.ln_2.parameters());
        out.extend(self.mlp.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.ln_1.parameters_mut());
        out.extend(self.attn.parameters_mut());
        out.extend(self.ln_2.parameters_mut());
        out.extend(self.mlp.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (name, param) in self.ln_1.named_parameters() {
            out.push(("ln_1.".to_string() + &name, param));
        }
        for (name, param) in self.attn.named_parameters() {
            out.push(("attn.".to_string() + &name, param));
        }
        for (name, param) in self.ln_2.named_parameters() {
            out.push(("ln_2.".to_string() + &name, param));
        }
        for (name, param) in self.mlp.named_parameters() {
            out.push(("mlp.".to_string() + &name, param));
        }
        out
    }

    fn train(&mut self) {
        todo!()
    }

    fn eval(&mut self) {
        todo!()
    }

    fn is_training(&self) -> bool {
        todo!()
    }

    fn named_buffers(&self) -> Vec<(String, &Buffer<T>)> {
        vec![]
    }

    fn buffers_mut(&mut self) -> Vec<&mut Buffer<T>> {
        let mut out = Vec::new();
        out.extend(self.ln_1.buffers_mut());
        out.extend(self.attn.buffers_mut());
        out.extend(self.ln_2.buffers_mut());
        out.extend(self.mlp.buffers_mut());
        out
    }
}

struct MLP<T: Float> {
    c_fc: Linear<T>,
    gelu: GELU,
    c_proj: Linear<T>,
}

impl<T: Float> MLP<T> {
    fn new(config: GPTConfig) -> Self {
        Self {
            c_fc: Linear::new(config.n_embd, config.n_embd * 4, true).unwrap(),
            gelu: GELU::with_approximate(GeluApproximate::Tanh),
            c_proj: Linear::new(config.n_embd * 4, config.n_embd, true).unwrap(),
        }
    }
}

impl<T: Float> Module<T> for MLP<T> {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let x: Tensor<T> = self.c_fc.forward(&x)?;
        let x: Tensor<T> = self.gelu.forward(&x)?;
        let x: Tensor<T> = self.c_proj.forward(&x)?;

        Ok(x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.c_fc.parameters());
        out.extend(self.c_proj.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.c_fc.parameters_mut());
        out.extend(self.c_proj.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (name, param) in self.c_fc.named_parameters() {
            out.push(("c_fc.".to_string() + &name, param));
        }
        for (name, param) in self.c_proj.named_parameters() {
            out.push(("c_proj.".to_string() + &name, param));
        }
        out
    }

    fn train(&mut self) {
        todo!()
    }

    fn eval(&mut self) {
        todo!()
    }

    fn is_training(&self) -> bool {
        todo!()
    }

    fn named_buffers(&self) -> Vec<(String, &Buffer<T>)> {
        vec![]
    }

    fn buffers_mut(&mut self) -> Vec<&mut Buffer<T>> {
        vec![]
    }
}

struct CausalSelfAttention<T: Float> {
    c_attn: Linear<T>,
    c_proj: Linear<T>,

    n_head: usize,
    n_embd: usize,

    // More of a "mask"
    bias: Buffer<T>,
}

impl<T: Float> CausalSelfAttention<T> {
    fn new(config: GPTConfig) -> Self {
        let bs = config.block_size;
        let mut mask_data = vec![<T as ferrotorch::Element>::zero(); bs * bs];
        for i in 0..bs {
            for j in (i + 1)..bs {
                mask_data[i * bs + j] = T::neg_infinity();
            }
        }
        let mask = from_vec(mask_data, &[1, 1, bs, bs]).unwrap();

        Self {
            c_attn: Linear::new(config.n_embd, config.n_embd * 3, true).unwrap(),
            c_proj: Linear::new(config.n_embd, config.n_embd, true).unwrap(),
            n_head: config.n_head,
            n_embd: config.n_embd,
            bias: Buffer::new(mask), // TODO unwrap err?
        }
    }
}

impl<T: Float> Module<T> for CausalSelfAttention<T> {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let [b, t, c] = x.shape() else { todo!() }; // n_embed is C

        let qkv: Tensor<T> = self.c_attn.forward(&x)?;
        let [q, k, v]: [Tensor<T>; 3] =
            qkv.split(&[*c, *c, *c], 2)?
                .try_into()
                .map_err(|v: Vec<Tensor<T>>| FerrotorchError::ShapeMismatch {
                    message: format!("expected 3 tensors from split, got {}", v.len()),
                })?;

        let k = k
            .contiguous()?
            .view(&[
                *b as i64,
                *t as i64,
                self.n_head as i64,
                (self.n_embd / self.n_head) as i64,
            ])?
            .transpose(1, 2)?
            .contiguous()?;
        let q = q
            .contiguous()?
            .view(&[
                *b as i64,
                *t as i64,
                self.n_head as i64,
                (self.n_embd / self.n_head) as i64,
            ])?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .contiguous()?
            .view(&[
                *b as i64,
                *t as i64,
                self.n_head as i64,
                (self.n_embd / self.n_head) as i64,
            ])?
            .transpose(1, 2)?
            .contiguous()?;

        let krank = k.ndim();
        let kt = k.transpose(krank - 2, krank - 1)?.contiguous()?;
        let attn = q.matmul(&kt)?;
        let scale = <T as ferrotorch::Element>::one()
            / T::from((self.n_embd / self.n_head) as f64).unwrap().sqrt();
        let attn = (&attn * &scalar(scale)?.to(attn.device())?)?;

        let mask_slice = self
            .bias
            .narrow(2, 0, *t as usize)?
            .narrow(3, 0, *t as usize)?
            .contiguous()?;

        let mask_full = expand(
            &mask_slice,
            &[*b as usize, self.n_head as usize, *t as usize, *t as usize],
        )?
        .contiguous()?;
        let attn = attn.add_t(&mask_full)?;

        let attn = attn.softmax()?;

        let y = attn.matmul(&v)?;
        let y = y.transpose(1, 2)?.contiguous()?.reshape_t(&[
            (*b).try_into().unwrap(),
            (*t).try_into().unwrap(),
            (*c).try_into().unwrap(),
        ])?;

        let y = self.c_proj.forward(&y)?;

        Ok(y)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.c_attn.parameters());
        out.extend(self.c_proj.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        // Same order as named_parameters
        let mut out = Vec::new();
        out.extend(self.c_attn.parameters_mut());
        out.extend(self.c_proj.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (name, param) in self.c_attn.named_parameters() {
            out.push((format!("c_attn.{name}"), param))
        }
        for (name, param) in self.c_proj.named_parameters() {
            out.push((format!("c_proj.{name}"), param))
        }

        out
    }

    fn buffers_mut(&mut self) -> Vec<&mut Buffer<T>> {
        let mut out = Vec::new();
        out.push(&mut self.bias);
        out
    }

    fn named_buffers(&self) -> Vec<(String, &Buffer<T>)> {
        let mut out = Vec::new();
        out.push(("bias".to_string(), &self.bias));
        out
    }

    fn train(&mut self) {
        todo!()
    }

    fn eval(&mut self) {
        todo!()
    }

    fn is_training(&self) -> bool {
        todo!()
    }
}

fn cuda_available() -> bool {
    // Unsafe eek
    // but avoids unwinding a panic
    unsafe {
        ["libcuda.so.1", "libcuda.so"]
            .iter()
            .any(|n| libloading::Library::new(*n).is_ok())
    }
}
