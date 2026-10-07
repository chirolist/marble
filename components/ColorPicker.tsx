"use client";

import { SWATCHES, hexToRgb, type RGB } from "@/lib/fluid/palette";

type Props = {
  selected: string;
  custom: string;
  onSelect: (hex: string, rgb: RGB) => void;
};

export function ColorPicker({ selected, custom, onSelect }: Props) {
  const customSelected = selected === custom && !SWATCHES.some((swatch) => swatch.hex === selected);

  return (
    <div className="marble-palette" role="group" aria-label="インクの色">
      {SWATCHES.map((swatch) => (
        <button
          key={swatch.id}
          type="button"
          className="marble-swatch"
          style={{ backgroundColor: swatch.hex }}
          aria-label={swatch.name}
          aria-pressed={selected === swatch.hex}
          onClick={() => onSelect(swatch.hex, swatch.rgb)}
        />
      ))}
      <label className="marble-custom" data-pressed={customSelected} aria-label="自由な色">
        <span style={{ backgroundColor: custom }} />
        <input
          type="color"
          value={custom}
          aria-label="自由な色を選ぶ"
          onChange={(event) => onSelect(event.target.value, hexToRgb(event.target.value))}
        />
      </label>
    </div>
  );
}
