use anyhow::{Context, Error, Result};
use hf_hub::api::sync::ApiBuilder;
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    model::{params::LlamaModelParams, LlamaModel},
};
use std::{num::NonZeroU32, sync::Arc};

pub struct ModelConfig {
    pub n_gpu_layers: i32,
    pub context_size: u32,
    pub model_repo: String,
    pub model_file: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            n_gpu_layers: 1000,
            context_size: 2048,
            model_repo: "lmstudio-community/zeta-GGUF".to_string(),
            model_file: "zeta-Q8_0.gguf".to_string(),
        }
    }
}

pub struct ModelManager {
    pub model: LlamaModel,
    pub context: LlamaContextParams,
}

impl ModelManager {
    pub fn new(config: ModelConfig, backend: Arc<LlamaBackend>) -> Result<Self, Error> {
        let model_params =
            LlamaModelParams::default().with_n_gpu_layers(config.n_gpu_layers.try_into().unwrap());
        println!("Loading model from {}", config.model_repo);
        let model_path = ApiBuilder::new()
            .with_progress(true)
            .build()
            .with_context(|| "unable to create huggingface api")?
            .model(config.model_repo)
            .get(&config.model_file)
            .with_context(|| "unable to download model")?;
        println!("Model downloaded");
        let model = LlamaModel::load_from_file(&backend.clone(), model_path, &model_params)
            .with_context(|| "unable to load model")?;

        dbg!("{:?}", model.n_layer());
        let context = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(config.context_size).unwrap()));

        Ok(Self { model, context })
    }
}
