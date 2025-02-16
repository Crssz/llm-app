import { invoke } from "@tauri-apps/api/core";
import "./App.css";
import { useState } from "react";
import Page from "./app/dashboard/page";

function App() {
  const [prompt, setPrompt] = useState("");
  const handleGenerate = async () => {
    const result = await invoke("generate_text", { prompt }).catch(
      console.error
    );
    console.log(result);
  };
  const handleSwitchModel = async () => {
    const result = await invoke("switch_model");
    console.log(result);
  };
  return <Page />;
}

export default App;
