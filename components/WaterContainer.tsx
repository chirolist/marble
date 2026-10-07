"use client";

import { forwardRef } from "react";

type Props = {
  error: string | null;
  onPointerDown: (event: React.PointerEvent<HTMLCanvasElement>) => void;
};

export const WaterContainer = forwardRef<HTMLCanvasElement, Props>(function WaterContainer(
  { error, onPointerDown },
  ref,
) {
  return (
    <div className="marble-stage-wrap">
      <div className="marble-stage">
        <canvas
          ref={ref}
          className="marble-canvas"
          role="application"
          aria-label="水の入ったガラス容器。タップすると、選んだ色の水滴が落ちます"
          onPointerDown={onPointerDown}
        />
        <div className="marble-glass" aria-hidden="true" />
        {error ? <p className="marble-error">{error}</p> : null}
      </div>
    </div>
  );
});
