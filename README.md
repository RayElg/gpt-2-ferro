# Gpt-2-ferro

Mostly faithful reproduction of Karpathy's ["Let's reproduce GPT-2"](https://www.youtube.com/watch?v=l8pRSuU81PU) using [Ferrotorch](https://github.com/forecast-bio/ferrotorch). ``Brain-coded`` as a learning opportunity.

There is basically parity between pytorch & ferrotorch, so the machinery between the pytorch & ferrotorch implementation are basically the same. The main difference that leads to different outputs is the algo for token sampling (& lack of seeding).

## Usage

Decodes 5 batches of 30 new tokens
```sh
cargo run -- "Once upon a time"
```

## Next

TODO - Execute on CUDA (or just non-CPU device)  
TODO - Training  
TODO - "Let's make it fast"  