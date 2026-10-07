type Props = {
  onReset: () => void;
};

export function GameControls({ onReset }: Props) {
  return (
    <div className="marble-controls">
      <button type="button" className="marble-reset" onClick={onReset}>
        リセット
      </button>
    </div>
  );
}
