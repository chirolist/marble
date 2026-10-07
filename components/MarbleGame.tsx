"use client";

import { useEffect, useRef, useState } from "react";
import { ColorPicker } from "@/components/ColorPicker";
import { GameControls } from "@/components/GameControls";
import { WaterContainer } from "@/components/WaterContainer";
import { FluidSimulation } from "@/lib/fluid/FluidSimulation";
import { SWATCHES, type RGB } from "@/lib/fluid/palette";

export function MarbleGame() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const simRef = useRef<FluidSimulation | null>(null);
  const colorRef = useRef<RGB>(SWATCHES[0].rgb);
  const [selected, setSelected] = useState(SWATCHES[0].hex);
  const [custom, setCustom] = useState("#e23d78");
  const [showHint, setShowHint] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let sim: FluidSimulation;
    try {
      sim = new FluidSimulation(canvas, {
        color: colorRef.current,
        onError: (message) => setError(message),
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "シミュレーションを開始できませんでした");
      return;
    }

    simRef.current = sim;
    sim.start();
    return () => {
      sim.dispose();
      if (simRef.current === sim) simRef.current = null;
    };
  }, []);

  const choose = (hex: string, rgb: RGB) => {
    colorRef.current = rgb;
    setSelected(hex);
    if (!SWATCHES.some((swatch) => swatch.hex === hex)) {
      setCustom(hex);
    }
    simRef.current?.setColor(rgb);
  };

  const dropAt = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const sim = simRef.current;
    if (!canvas || !sim || error) return;
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) / Math.max(1, rect.width);
    sim.spawnDrop(x);
    setShowHint(false);
  };

  return (
    <main className="marble">
      <header className="marble-header">
        <h1>マーブル</h1>
        <p className={showHint ? "marble-hint" : "marble-hint is-hidden"}>水面をタップ</p>
      </header>
      <ColorPicker selected={selected} custom={custom} onSelect={choose} />
      <WaterContainer ref={canvasRef} error={error} onPointerDown={dropAt} />
      <GameControls
        onReset={() => {
          simRef.current?.reset();
        }}
      />
    </main>
  );
}
