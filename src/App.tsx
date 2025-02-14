import { invoke } from "@tauri-apps/api/core";
import "./App.css";
import { useState } from "react";

function App() {
  const [prompt, setPrompt] = useState("");
  const handleGenerate = async () => {
    const result = await invoke("generate_text", { prompt });
    console.log(result);
  };
  const handleSwitchModel = async () => {
    const result = await invoke("switch_model");
    console.log(result);
  };
  return (
    <main className="container">
      <h1>Welcome to Tauri + React</h1>
      <input
        type="text"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
      />
      <button onClick={handleGenerate}>Generate</button>
      <button onClick={handleSwitchModel}>Switch Model</button>
    </main>
  );
}

export default App;
