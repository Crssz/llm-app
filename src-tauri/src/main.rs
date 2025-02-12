// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::PathBuf;

use candle_core::DType;
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{Config as ConfigBase, ModelForCausalLM as ModelBase};
use hf_hub::{api::sync::Api, Repo, RepoType};
use rag_llm_app_lib::models::{self, TextGeneration};
use tokenizers::Tokenizer;

fn main() -> Result<(), anyhow::Error> {
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let model_id = format!("Qwen/Qwen2-7b");
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        "main".to_string(),
    ));

    let config_file = repo.get("config.json")?;
    let config: ConfigBase = serde_json::from_slice(&std::fs::read(config_file)?)?;

    // model-00001-of-00008.safetensors

    let tokenizer_filename = repo.get("tokenizer.json")?;

    let filenames: Vec<PathBuf> = {
        let mut filenames = Vec::new();
        if config.num_key_value_heads == 1 {
            filenames.push(repo.get("model.safetensors")?);
        } else {
            for i in 1..=config.num_key_value_heads {
                let filename = format!(
                    "model-0000{}-of-0000{}.safetensors",
                    i, config.num_key_value_heads
                );
                println!("loading the file: {}", filename);
                let path = repo.get(&filename)?;
                println!("done..");
                filenames.push(path);
            }
        }
        filenames
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(candle_core::Error::msg)?;

    let start: std::time::Instant = std::time::Instant::now();

    let device = candle_core::Device::new_cuda(0)?;
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    let model = models::Model::Base(ModelBase::new(&config, vb)?);

    println!("loaded the model in {:?}", start.elapsed());

    let mut pipeline =
        TextGeneration::new(model, tokenizer, 299792458, None, None, 1.1, 64, &device);
    pipeline.run("imoplement basic python algebra", 10000)?;
    rag_llm_app_lib::run();
    Ok(())
}
