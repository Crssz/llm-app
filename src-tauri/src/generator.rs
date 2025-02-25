use anyhow::{bail, Context as _, Result};
use llama_cpp_2::{
    context::LlamaContext,
    ggml_time_us,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, Special},
    sampling::LlamaSampler,
    token::LlamaToken,
};
use std::{io::Write, sync::Arc, time::Duration};

use crate::model::ModelManager;

pub struct GenerationConfig {
    pub max_length: i32,
    pub batch_size: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_length: 4096 * 2,
            batch_size: 512,
        }
    }
}

pub struct TextGenerator {
    model_manager: Arc<ModelManager>,
    config: GenerationConfig,
}

impl TextGenerator {
    pub fn new(model_manager: Arc<ModelManager>, config: GenerationConfig) -> Self {
        Self {
            model_manager,
            config,
        }
    }

    pub fn generate(&self, prompt: &str, backend: Arc<LlamaBackend>, callback: Option<&dyn Fn(String)>) -> Result<()> {
        let mut ctx = self
            .model_manager
            .model
            .new_context(&backend, self.model_manager.context.clone())
            .with_context(|| "unable to create the llama_context")?;

        let tokens_list = self
            .model_manager
            .model
            .str_to_token(prompt, AddBos::Always)
            .with_context(|| format!("failed to tokenize {prompt}"))?;

        self.validate_context_size(&tokens_list)?;
        self.print_tokens(&tokens_list)?;

        let mut batch = LlamaBatch::new(self.config.batch_size, 1);
        self.prepare_initial_batch(&mut batch, &tokens_list)?;

        ctx.decode(&mut batch)
            .with_context(|| "llama_decode() failed")?;

        self.generate_text(&mut ctx, &mut batch, callback)?;

        Ok(())
    }

    fn validate_context_size(&self, tokens_list: &[LlamaToken]) -> Result<()> {
        let n_len = self.config.max_length;
        let n_cxt = self.model_manager.context.n_ctx().unwrap().get() as i32;
        let n_kv_req = tokens_list.len() as i32 + (n_len - tokens_list.len() as i32);

        eprintln!("n_len = {n_len}, n_ctx = {n_cxt}, k_kv_req = {n_kv_req}");

        if n_kv_req > n_cxt {
            bail!(
                "n_kv_req > n_ctx, the required kv cache size is not big enough
either reduce n_len or increase n_ctx"
            )
        }

        if tokens_list.len() >= usize::try_from(n_len)? {
            bail!("the prompt is too long, it has more tokens than n_len")
        }

        Ok(())
    }

    fn print_tokens(&self, tokens_list: &[LlamaToken]) -> Result<()> {
        eprintln!();
        for token in tokens_list {
            eprint!(
                "{}",
                self.model_manager
                    .model
                    .token_to_str(*token, Special::Tokenize)?
            );
        }
        std::io::stderr().flush()?;
        Ok(())
    }

    fn prepare_initial_batch(
        &self,
        batch: &mut LlamaBatch,
        tokens_list: &[LlamaToken],
    ) -> Result<()> {
        let last_index: i32 = (tokens_list.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens_list.iter()) {
            let is_last = i == last_index;
            batch.add(*token, i, &[0], is_last)?;
        }
        Ok(())
    }

    fn generate_text(
        &self,
        ctx: &mut LlamaContext,
        batch: &mut LlamaBatch,
        callback: Option<&dyn Fn(String)>,
    ) -> Result<()> {
        let mut n_cur = batch.n_tokens();
        let mut n_decode = 0;
        let t_main_start = ggml_time_us();

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut sampler =
            LlamaSampler::chain_simple([LlamaSampler::dist(1234), LlamaSampler::greedy()]);

        while n_cur <= self.config.max_length {
            let token = sampler.sample(ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if self.model_manager.model.is_eog_token(token) {
                eprintln!();
                break;
            }

            let output_bytes = self
                .model_manager
                .model
                .token_to_bytes(token, Special::Tokenize)?;
            let mut output_string = String::with_capacity(32);
            let _decode_result = decoder.decode_to_string(&output_bytes, &mut output_string, false);
            print!("{output_string}");
            if let Some(callback) = callback {
                callback(output_string);
            }
            std::io::stdout().flush()?;

            batch.clear();
            batch.add(token, n_cur, &[0], true)?;

            n_cur += 1;
            ctx.decode(batch).with_context(|| "failed to eval")?;
            n_decode += 1;
        }

        self.print_stats(t_main_start, n_decode)?;
        println!("{}", ctx.timings());

        Ok(())
    }

    fn print_stats(&self, t_main_start: i64, n_decode: i32) -> Result<()> {
        eprintln!("\n");
        let t_main_end = ggml_time_us();
        let duration = Duration::from_micros((t_main_end - t_main_start) as u64);

        eprintln!(
            "decoded {} tokens in {:.2} s, speed {:.2} t/s\n",
            n_decode,
            duration.as_secs_f32(),
            n_decode as f32 / duration.as_secs_f32()
        );
        Ok(())
    }
}
