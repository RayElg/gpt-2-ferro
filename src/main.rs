use std::collections::HashMap;

use ferrotorch::{
    distributions::{Categorical, Distribution},
    expand, from_vec,
    hub::{HubCache, hf_download_model},
    nn::{Buffer, Embedding, ModuleList, StateDict},
    no_grad,
    prelude::*,
    serialize::load_safetensors,
    tokenize::{Tokenizer, decode, encode, load_tokenizer},
};

fn main() -> FerrotorchResult<()> {
    let in_str = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Hello, I'm a language model,".to_string());

    let res = do_gpt2(&in_str);
    println!("{res:?}");
    Ok(())
}

fn do_gpt2(in_str: &str) -> FerrotorchResult<()> {
    let cache = HubCache::with_default_dir();
    let dir = hf_download_model("openai-community/gpt2", "main", &cache)?;

    println!("Getting weights");
    let state_dict = load_safetensors::<f32>(&dir.join("model.safetensors"))?;
    let tok = load_tokenizer(&dir.join("tokenizer.json"))?;

    let mut gpt = GPT::new(GPTConfig::GPT2);

    println!("Loading weights into gpt");
    let gpt = gpt.load_from_statedict(&state_dict)?;
    let gpt = gpt.set_tok(tok)?;

    let size_batch = 5;
    let decode_n_toks = 30;

    let out = no_grad(|| gpt.pipeline(in_str, size_batch, decode_n_toks))?;

    for s in out {
        println!("> {}", s);
    }

    Ok(())
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

struct GPT<T: Float> {
    config: GPTConfig,
    transformer: Transformer<T>,
    lm_head: Linear<T>,
    tok: Option<Tokenizer>,
}

impl<T: Float> GPT<T> {
    fn new(config: GPTConfig) -> Self {
        Self {
            config,
            transformer: Transformer::new(config),
            lm_head: Linear::new(config.n_embd, config.vocab_size, false).unwrap(),
            tok: None,
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
        )?;
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
        let mut batch: Tensor<T> = from_vec(data, &[size_batch, t])?;

        while batch.size()[1] < (t + decode_n_toks) {
            let logits = &self.forward(&batch)?;
            let cur_t = batch.size()[1];
            let logits = logits.narrow(1, cur_t - 1, 1)?.contiguous()?.squeeze_t(1)?;
            let probs = logits.softmax()?;
            let (topk_probs, topk_indices) = topk(&probs, 50, true)?;

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
            let new_col = from_vec(new_col_data, &[b, 1])?;

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

struct Transformer<T: Float> {
    wte: Embedding<T>,
    wpe: Embedding<T>,
    h: ModuleList<T>,
    ln_f: LayerNorm<T>,
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
        Self {
            ln_1: LayerNorm::new(vec![config.n_embd], 1e-5, true).unwrap(), // TODO check learnable?
            attn: CausalSelfAttention::new(config),
            ln_2: LayerNorm::new(vec![config.n_embd], 1e-5, true).unwrap(), // TODO check learnable?
            mlp: MLP::new(config),
        }
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
        todo!()
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
        vec![]
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
        todo!()
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
        let ones_matrix = ones::<T>(&[config.block_size, config.block_size]).unwrap();
        let lower_tri = ferrotorch::tril(&ones_matrix, 0).unwrap();
        let mask = lower_tri
            .view(&[1, 1, config.block_size as i64, config.block_size as i64])
            .unwrap();

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
        let attn = (&attn * &scalar(scale)?)?;

        let mask_slice = self
            .bias
            .narrow(2, 0, *t as usize)?
            .narrow(3, 0, *t as usize)?
            .contiguous()?;
        let mask_expanded = expand(
            &mask_slice,
            &[*b as usize, self.n_head as usize, *t as usize, *t as usize],
        )?
        .contiguous()?;
        let bool_mask = BoolTensor::from_predicate(&mask_expanded, |v| {
            v == <T as ferrotorch::Element>::zero()
        })?;
        let attn = attn.masked_fill(&bool_mask, T::neg_infinity())?;

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
        todo!()
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
