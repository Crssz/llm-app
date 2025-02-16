use generator::{GenerationConfig, TextGenerator};
use llama_cpp_2::llama_backend::LlamaBackend;
use model::{ModelConfig, ModelManager};
use std::{error::Error, sync::Arc};
use tauri::{async_runtime::Mutex, AppHandle, Manager, State};

mod generator;
mod model;

struct AppState {
    backend: Arc<LlamaBackend>,
    generator: Arc<TextGenerator>,
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![generate_text, switch_model])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[tauri::command]
async fn generate_text(prompt: String, state: State<'_, Mutex<AppState>>) -> Result<(), ()> {
    let state = state.lock().await;
    state
        .generator
        .generate(&prompt, state.backend.clone())
        .unwrap();
    Ok(())
}

#[tauri::command]
async fn switch_model(state: State<'_, Mutex<AppState>>) -> Result<(), ()> {
    let mut state = state.lock().await;
    let backend = state.backend.clone();
    let model_config = ModelConfig {
        n_gpu_layers: 1,
        context_size: 1024,
        model_repo: "lmstudio-community/Qwen2.5-Coder-7B-Instruct-GGUF".to_string(),
        model_file: "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf".to_string(),
    };
    let model_manager = Arc::new(ModelManager::new(model_config, backend.clone()).unwrap());
    let gen_config = GenerationConfig::default();
    let generator = Arc::new(TextGenerator::new(model_manager.clone(), gen_config));

    state.generator = generator.clone();

    Ok(())
}
