// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod generator;
mod model;

use anyhow::Result;


fn main() -> Result<()> {
    rag_llm_app_lib::run();
    Ok(())
}
