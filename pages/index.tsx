import Head from "next/head";
import Yolo from "../components/models/Yolo";
import YoloServer from "../components/models/YoloServer";
import AlprServer from "../components/models/AlprServer";
import StreamServer from "../components/models/StreamServer";
import { useState } from "react";

export default function Home() {
  const [mode, setMode] = useState<"local" | "server" | "alpr" | "stream">(
    "local"
  );

  return (
    <>
      <main className="font-mono flex flex-col justify-center items-center  w-screen">
        <h1 className="m-5 text-xl font-bold">Real-Time Object Detection</h1>
        <div className="flex gap-1 mb-3">
          <button
            onClick={() => setMode("local")}
            className={`p-2 border-2 border-dashed rounded-xl hover:translate-y-1 ${
              mode === "local" ? "bg-white text-black" : ""
            }`}
          >
            In-browser (WASM)
          </button>
          <button
            onClick={() => setMode("server")}
            className={`p-2 border-2 border-dashed rounded-xl hover:translate-y-1 ${
              mode === "server" ? "bg-white text-black" : ""
            }`}
          >
            Server-side
          </button>
          <button
            onClick={() => setMode("alpr")}
            className={`p-2 border-2 border-dashed rounded-xl hover:translate-y-1 ${
              mode === "alpr" ? "bg-white text-black" : ""
            }`}
          >
            License plates (server, ~1 fps)
          </button>
          <button
            onClick={() => setMode("stream")}
            className={`p-2 border-2 border-dashed rounded-xl hover:translate-y-1 ${
              mode === "stream" ? "bg-white text-black" : ""
            }`}
          >
            Full-res stream (server, 1-6 fps)
          </button>
        </div>
        {mode === "local" ? (
          <Yolo />
        ) : mode === "server" ? (
          <YoloServer />
        ) : mode === "alpr" ? (
          <AlprServer />
        ) : (
          <StreamServer />
        )}
        <p className="m-5">
          Created by{" "}
          <a
            className="underline underline-offset-1 hover:translate-y-1"
            href="https://juanjaho.github.io/"
          >
            @juanjaho
          </a>
        </p>
      </main>
    </>
  );
}
