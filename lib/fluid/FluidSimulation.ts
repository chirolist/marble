import { pigmentAbsorption, type RGB } from "./palette";
import {
  ADVECT_FRAG,
  CURL_FRAG,
  DIFFUSE_FRAG,
  DISPLAY_FRAG,
  DIVERGENCE_FRAG,
  FORCES_FRAG,
  GRADIENT_FRAG,
  MASK_FRAG,
  PRESSURE_FRAG,
  SPLAT_FRAG,
  VERT,
  VORTEX_FRAG,
} from "./shaders";
import {
  AMBIENT,
  BUOYANCY,
  CONFINEMENT,
  DYE_DECAY,
  DYE_DIFFUSE,
  GRAVITY,
  MAX_DROPS,
  MAX_RIPPLES,
  MAX_SPARKS,
  MAX_SPLATS,
  MAX_VORTICES,
  MENISCUS,
  PRESSURE_ITERS,
  SIM_H,
  SIM_W,
  SURFACE,
  VEL_DECAY,
  WAVE_ENCODE,
  WAVE_N,
} from "./tuning";

type TexFBO = {
  tex: WebGLTexture;
  fbo: WebGLFramebuffer;
};

type DoubleFBO = {
  read: TexFBO;
  write: TexFBO;
  swap: () => void;
};

type Drop = {
  x: number;
  y: number;
  vy: number;
  rgb: RGB;
  absorption: RGB;
  age: number;
  squash: number;
  state: "fall" | "squash";
  seed: number;
};

type Spark = {
  x: number;
  y: number;
  vx: number;
  vy: number;
  age: number;
  life: number;
  rgb: RGB;
  radius: number;
};

type Ripple = {
  x: number;
  age: number;
  strength: number;
};

type DyeSplat = {
  x: number;
  y: number;
  radius: number;
  scale: number;
  stretchX: number;
  stretchY: number;
  absorption: RGB;
};

type Vortex = {
  x: number;
  y: number;
  sep: number;
  radius: number;
  strength: number;
  jet: number;
  spin: number;
  spread: number;
};

export type FluidOptions = {
  color: RGB;
  onError?: (message: string) => void;
};

const TEXEL: [number, number] = [1 / SIM_W, 1 / SIM_H];

export class FluidSimulation {
  private readonly canvas: HTMLCanvasElement;
  private readonly gl: WebGL2RenderingContext;
  private readonly onError?: (message: string) => void;

  private readonly vao: WebGLVertexArrayObject;
  private readonly programs: Record<string, WebGLProgram>;
  private readonly locations = new Map<string, WebGLUniformLocation | null>();

  private readonly velocity: DoubleFBO;
  private readonly dye: DoubleFBO;
  private readonly pressure: DoubleFBO;
  private readonly divergence: TexFBO;
  private readonly curl: TexFBO;
  private readonly waveTex: WebGLTexture;
  private readonly waveBytes = new Uint8Array(WAVE_N * 4);

  private readonly h = new Float32Array(WAVE_N);
  private readonly v = new Float32Array(WAVE_N);

  private readonly drops: Drop[] = [];
  private readonly pendingDrops: Array<{ x: number; rgb: RGB }> = [];
  private readonly sparks: Spark[] = [];
  private readonly ripples: Ripple[] = [];
  private readonly dyeSplats: DyeSplat[] = [];
  private readonly vortices: Vortex[] = [];

  private readonly dropData = new Float32Array(MAX_DROPS * 4);
  private readonly dropColors = new Float32Array(MAX_DROPS * 3);
  private readonly dropMeta = new Float32Array(MAX_DROPS * 4);
  private readonly sparkData = new Float32Array(MAX_SPARKS * 4);
  private readonly sparkColors = new Float32Array(MAX_SPARKS * 3);
  private readonly rippleData = new Float32Array(MAX_RIPPLES * 4);
  private readonly splatS0 = new Float32Array(MAX_SPLATS * 4);
  private readonly splatS1 = new Float32Array(MAX_SPLATS * 4);
  private readonly splatC0 = new Float32Array(MAX_SPLATS * 3);
  private readonly vortexV0 = new Float32Array(MAX_VORTICES * 4);
  private readonly vortexV1 = new Float32Array(MAX_VORTICES * 4);

  private color: RGB;
  private time = 0;
  private raf = 0;
  private running = false;
  private disposed = false;
  private acc = 0;
  private last = 0;
  private frames = 0;
  private fpsStamp = 0;
  private resizeObserver: ResizeObserver | null = null;
  private internalFormat: number;
  private floatType: number;

  constructor(canvas: HTMLCanvasElement, options: FluidOptions) {
    this.canvas = canvas;
    this.color = options.color;
    this.onError = options.onError;

    const gl = canvas.getContext("webgl2", {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      premultipliedAlpha: false,
      powerPreference: "high-performance",
      failIfMajorPerformanceCaveat: false,
    });
    if (!gl) {
      throw new Error("WebGL2が使えないため、流体を描画できません");
    }
    this.gl = gl;
    gl.getExtension("EXT_color_buffer_float");
    gl.getExtension("EXT_color_buffer_half_float");
    gl.getExtension("OES_texture_float_linear");
    gl.getExtension("OES_texture_half_float_linear");

    const format = this.pickFloatFormat();
    if (!format) {
      throw new Error("この環境では流体シミュレーション用の浮動小数バッファが使えません");
    }
    this.internalFormat = format.internalFormat;
    this.floatType = format.type;

    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

    this.vao = this.createQuad();
    this.programs = {
      advect: this.link("advect", ADVECT_FRAG),
      curl: this.link("curl", CURL_FRAG),
      forces: this.link("forces", FORCES_FRAG),
      vortex: this.link("vortex", VORTEX_FRAG),
      divergence: this.link("divergence", DIVERGENCE_FRAG),
      pressure: this.link("pressure", PRESSURE_FRAG),
      gradient: this.link("gradient", GRADIENT_FRAG),
      diffuse: this.link("diffuse", DIFFUSE_FRAG),
      splat: this.link("splat", SPLAT_FRAG),
      mask: this.link("mask", MASK_FRAG),
      display: this.link("display", DISPLAY_FRAG),
    };

    this.velocity = this.doubleFBO();
    this.dye = this.doubleFBO();
    this.pressure = this.doubleFBO();
    this.divergence = this.singleFBO();
    this.curl = this.singleFBO();
    this.waveTex = this.createWaveTexture();

    canvas.addEventListener("webglcontextlost", (event) => {
      event.preventDefault();
      this.fail("描画が中断されました");
    });

    this.clearSimulation();
    this.resize();
    this.uploadWaves();
    this.render();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
  }

  start(): void {
    if (this.disposed || this.running) return;
    this.running = true;
    this.last = performance.now();
    this.fpsStamp = this.last;
    const loop = (now: number) => {
      if (!this.running || this.disposed) return;
      this.raf = requestAnimationFrame(loop);
      const frameDt = Math.min(0.05, (now - this.last) / 1000);
      this.last = now;
      this.acc += frameDt;
      const step = 1 / 60;
      let steps = 0;
      while (this.acc >= step && steps < 2) {
        this.step(step);
        this.acc -= step;
        steps += 1;
      }
      if (steps === 0) {
        this.render();
      }
      this.frames += 1;
      if (now - this.fpsStamp > 1000) {
        this.canvas.dataset.fps = String(Math.round((this.frames * 1000) / (now - this.fpsStamp)));
        this.frames = 0;
        this.fpsStamp = now;
      }
    };
    this.raf = requestAnimationFrame(loop);
  }

  dispose(): void {
    this.disposed = true;
    this.running = false;
    cancelAnimationFrame(this.raf);
    this.resizeObserver?.disconnect();
  }

  setColor(rgb: RGB): void {
    this.color = rgb;
  }

  spawnDrop(xNorm: number): void {
    const x = clamp(xNorm, 0.07, 0.93);
    const rgb = this.color;
    if (this.drops.length >= MAX_DROPS) {
      if (this.pendingDrops.length < 24) this.pendingDrops.push({ x, rgb });
      return;
    }
    this.pushDrop(x, rgb);
  }

  reset(): void {
    this.drops.length = 0;
    this.pendingDrops.length = 0;
    this.sparks.length = 0;
    this.ripples.length = 0;
    this.dyeSplats.length = 0;
    this.vortices.length = 0;
    this.h.fill(0);
    this.v.fill(0);
    this.clearSimulation();
    this.uploadWaves();
    this.render();
  }

  private fail(message: string): void {
    if (this.disposed) return;
    this.running = false;
    cancelAnimationFrame(this.raf);
    this.onError?.(message);
  }

  private pickFloatFormat(): { internalFormat: number; type: number } | null {
    const gl = this.gl;
    const candidates: Array<{ internalFormat: number; type: number }> = [
      { internalFormat: gl.RGBA16F, type: gl.HALF_FLOAT },
      { internalFormat: gl.RGBA32F, type: gl.FLOAT },
    ];
    for (const candidate of candidates) {
      if (this.supportsTarget(candidate.internalFormat, candidate.type)) {
        return candidate;
      }
    }
    return null;
  }

  private supportsTarget(internalFormat: number, type: number): boolean {
    const gl = this.gl;
    const tex = gl.createTexture();
    const fbo = gl.createFramebuffer();
    if (!tex || !fbo) return false;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, 4, 4, 0, gl.RGBA, type, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteTexture(tex);
    gl.deleteFramebuffer(fbo);
    gl.getError();
    return ok;
  }

  private createQuad(): WebGLVertexArrayObject {
    const gl = this.gl;
    const vao = gl.createVertexArray();
    const buffer = gl.createBuffer();
    if (!vao || !buffer) throw new Error("描画用の頂点バッファを作れませんでした");
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    return vao;
  }

  private link(name: string, source: string): WebGLProgram {
    const gl = this.gl;
    const vs = this.compile(gl.VERTEX_SHADER, VERT, name);
    const fs = this.compile(gl.FRAGMENT_SHADER, source, name);
    const program = gl.createProgram();
    if (!program) throw new Error(`${name}: program`);
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.bindAttribLocation(program, 0, "aPosition");
    gl.linkProgram(program);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(program) ?? "";
      throw new Error(`${name} のリンクに失敗しました: ${log}`);
    }
    return program;
  }

  private compile(type: number, source: string, name: string): WebGLShader {
    const gl = this.gl;
    const shader = gl.createShader(type);
    if (!shader) throw new Error(`${name}: shader`);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(shader) ?? "";
      gl.deleteShader(shader);
      throw new Error(`${name} のコンパイルに失敗しました: ${log}`);
    }
    return shader;
  }

  private createTarget(w: number, h: number): TexFBO {
    const gl = this.gl;
    const tex = gl.createTexture();
    const fbo = gl.createFramebuffer();
    if (!tex || !fbo) throw new Error("シミュレーション用のバッファを作れませんでした");
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, this.internalFormat, w, h, 0, gl.RGBA, this.floatType, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`フレームバッファが不完全です (${status})`);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { tex, fbo };
  }

  private singleFBO(): TexFBO {
    return this.createTarget(SIM_W, SIM_H);
  }

  private doubleFBO(): DoubleFBO {
    let read = this.createTarget(SIM_W, SIM_H);
    let write = this.createTarget(SIM_W, SIM_H);
    return {
      get read() {
        return read;
      },
      get write() {
        return write;
      },
      swap() {
        const tmp = read;
        read = write;
        write = tmp;
      },
    };
  }

  private createWaveTexture(): WebGLTexture {
    const gl = this.gl;
    const tex = gl.createTexture();
    if (!tex) throw new Error("水面テクスチャを作れませんでした");
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.waveBytes.fill(128);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, WAVE_N, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, this.waveBytes);
    return tex;
  }

  private clearFBO(target: TexFBO): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
    gl.viewport(0, 0, SIM_W, SIM_H);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
  }

  private clearSimulation(): void {
    this.clearFBO(this.velocity.read);
    this.clearFBO(this.velocity.write);
    this.clearFBO(this.dye.read);
    this.clearFBO(this.dye.write);
    this.clearFBO(this.pressure.read);
    this.clearFBO(this.pressure.write);
    this.clearFBO(this.divergence);
    this.clearFBO(this.curl);
  }

  private resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  private loc(program: WebGLProgram, name: string): WebGLUniformLocation | null {
    const key = `${programToId(program)}:${name}`;
    const cached = this.locations.get(key);
    if (cached !== undefined) return cached;
    const location = this.gl.getUniformLocation(program, name);
    this.locations.set(key, location);
    return location;
  }

  private bind(unit: number, tex: WebGLTexture, location: WebGLUniformLocation | null): void {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(location, unit);
  }

  private draw(program: WebGLProgram): void {
    const gl = this.gl;
    gl.useProgram(program);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  private targetSim(fbo: WebGLFramebuffer): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.viewport(0, 0, SIM_W, SIM_H);
  }

  private setSurfaceUniforms(program: WebGLProgram): void {
    const gl = this.gl;
    gl.uniform1f(this.loc(program, "uSurface"), SURFACE);
    gl.uniform1f(this.loc(program, "uTime"), this.time);
    gl.uniform1f(this.loc(program, "uWaveScale"), WAVE_ENCODE);
    gl.uniform1f(this.loc(program, "uMeniscus"), MENISCUS);
    gl.uniform2f(this.loc(program, "uTexel"), TEXEL[0], TEXEL[1]);
  }

  private step(dt: number): void {
    this.time += dt;
    this.stepDrops(dt);
    this.stepSparks(dt);
    this.stepWaves(dt);
    this.uploadWaves();

    this.advect(this.velocity, this.velocity, Math.exp(-VEL_DECAY * dt), false, dt);
    this.applyDyeSplats();
    this.computeCurl();
    this.applyForces(dt);
    this.applyVortices();
    this.computeDivergence();
    this.solvePressure();
    this.subtractGradient();
    this.advect(this.dye, this.velocity, Math.exp(-DYE_DECAY * dt), true, dt);
    this.diffuseDye(dt);
    this.maskDye();
    this.render();
  }

  private advect(
    field: DoubleFBO,
    velocity: DoubleFBO,
    dissipation: number,
    clampWater: boolean,
    dt: number,
  ): void {
    const gl = this.gl;
    const program = this.programs.advect;
    this.targetSim(field.write.fbo);
    gl.useProgram(program);
    this.bind(0, velocity.read.tex, this.loc(program, "uVelocity"));
    this.bind(1, field.read.tex, this.loc(program, "uSource"));
    this.bind(2, this.waveTex, this.loc(program, "uWave"));
    this.setSurfaceUniforms(program);
    gl.uniform1f(this.loc(program, "uDt"), dt);
    gl.uniform1f(this.loc(program, "uDissipation"), dissipation);
    gl.uniform1i(this.loc(program, "uClampWater"), clampWater ? 1 : 0);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    field.swap();
  }

  private computeCurl(): void {
    const gl = this.gl;
    const program = this.programs.curl;
    this.targetSim(this.curl.fbo);
    gl.useProgram(program);
    this.bind(0, this.velocity.read.tex, this.loc(program, "uVelocity"));
    gl.uniform2f(this.loc(program, "uTexel"), TEXEL[0], TEXEL[1]);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  private applyForces(dt: number): void {
    const gl = this.gl;
    const program = this.programs.forces;
    this.targetSim(this.velocity.write.fbo);
    gl.useProgram(program);
    this.bind(0, this.velocity.read.tex, this.loc(program, "uVelocity"));
    this.bind(1, this.dye.read.tex, this.loc(program, "uDye"));
    this.bind(2, this.curl.tex, this.loc(program, "uCurl"));
    this.bind(3, this.waveTex, this.loc(program, "uWave"));
    this.setSurfaceUniforms(program);
    gl.uniform1f(this.loc(program, "uDt"), dt);
    gl.uniform1f(this.loc(program, "uBuoyancy"), BUOYANCY);
    gl.uniform1f(this.loc(program, "uConfinement"), CONFINEMENT);
    gl.uniform1f(this.loc(program, "uAmbient"), AMBIENT);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    this.velocity.swap();
  }

  private applyVortices(): void {
    if (this.vortices.length === 0) return;
    const gl = this.gl;
    const program = this.programs.vortex;
    while (this.vortices.length > 0) {
      const batch = this.vortices.splice(0, MAX_VORTICES);
      this.vortexV0.fill(0);
      this.vortexV1.fill(0);
      for (let i = 0; i < batch.length; i++) {
        const v = batch[i];
        this.vortexV0.set([v.x, v.y, v.sep, v.radius], i * 4);
        this.vortexV1.set([v.strength, v.jet, v.spin, v.spread], i * 4);
      }
      this.targetSim(this.velocity.write.fbo);
      gl.useProgram(program);
      this.bind(0, this.velocity.read.tex, this.loc(program, "uVelocity"));
      gl.uniform2f(this.loc(program, "uTexel"), TEXEL[0], TEXEL[1]);
      gl.uniform1i(this.loc(program, "uCount"), batch.length);
      gl.uniform4fv(this.loc(program, "uV0[0]"), this.vortexV0);
      gl.uniform4fv(this.loc(program, "uV1[0]"), this.vortexV1);
      gl.bindVertexArray(this.vao);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      this.velocity.swap();
    }
  }

  private computeDivergence(): void {
    const gl = this.gl;
    const program = this.programs.divergence;
    this.targetSim(this.divergence.fbo);
    gl.useProgram(program);
    this.bind(0, this.velocity.read.tex, this.loc(program, "uVelocity"));
    this.bind(1, this.waveTex, this.loc(program, "uWave"));
    this.setSurfaceUniforms(program);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  private solvePressure(): void {
    this.clearFBO(this.pressure.read);
    this.clearFBO(this.pressure.write);
    const gl = this.gl;
    const program = this.programs.pressure;
    for (let i = 0; i < PRESSURE_ITERS; i++) {
      this.targetSim(this.pressure.write.fbo);
      gl.useProgram(program);
      this.bind(0, this.pressure.read.tex, this.loc(program, "uPressure"));
      this.bind(1, this.divergence.tex, this.loc(program, "uDivergence"));
      gl.uniform2f(this.loc(program, "uTexel"), TEXEL[0], TEXEL[1]);
      gl.bindVertexArray(this.vao);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      this.pressure.swap();
    }
  }

  private subtractGradient(): void {
    const gl = this.gl;
    const program = this.programs.gradient;
    this.targetSim(this.velocity.write.fbo);
    gl.useProgram(program);
    this.bind(0, this.pressure.read.tex, this.loc(program, "uPressure"));
    this.bind(1, this.velocity.read.tex, this.loc(program, "uVelocity"));
    this.bind(2, this.waveTex, this.loc(program, "uWave"));
    this.setSurfaceUniforms(program);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    this.velocity.swap();
  }

  private diffuseDye(dt: number): void {
    const gl = this.gl;
    const program = this.programs.diffuse;
    this.targetSim(this.dye.write.fbo);
    gl.useProgram(program);
    this.bind(0, this.dye.read.tex, this.loc(program, "uDye"));
    this.bind(1, this.waveTex, this.loc(program, "uWave"));
    this.setSurfaceUniforms(program);
    gl.uniform1f(this.loc(program, "uDt"), dt);
    gl.uniform1f(this.loc(program, "uDiffuse"), DYE_DIFFUSE);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    this.dye.swap();
  }

  private applyDyeSplats(): void {
    if (this.dyeSplats.length === 0) return;
    const gl = this.gl;
    const program = this.programs.splat;
    while (this.dyeSplats.length > 0) {
      const batch = this.dyeSplats.splice(0, MAX_SPLATS);
      this.splatS0.fill(0);
      this.splatS1.fill(0);
      this.splatC0.fill(0);
      for (let i = 0; i < batch.length; i++) {
        const s = batch[i];
        this.splatS0.set([s.x, s.y, s.radius, s.scale], i * 4);
        this.splatS1.set([s.stretchX, s.stretchY, 0, 0], i * 4);
        this.splatC0.set(s.absorption, i * 3);
      }
      this.targetSim(this.dye.write.fbo);
      gl.useProgram(program);
      this.bind(0, this.dye.read.tex, this.loc(program, "uTarget"));
      gl.uniform2f(this.loc(program, "uTexel"), TEXEL[0], TEXEL[1]);
      gl.uniform1i(this.loc(program, "uCount"), batch.length);
      gl.uniform4fv(this.loc(program, "uS0[0]"), this.splatS0);
      gl.uniform4fv(this.loc(program, "uS1[0]"), this.splatS1);
      gl.uniform3fv(this.loc(program, "uC0[0]"), this.splatC0);
      gl.bindVertexArray(this.vao);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      this.dye.swap();
    }
  }

  private maskDye(): void {
    const gl = this.gl;
    const program = this.programs.mask;
    this.targetSim(this.dye.write.fbo);
    gl.useProgram(program);
    this.bind(0, this.dye.read.tex, this.loc(program, "uDye"));
    this.bind(1, this.waveTex, this.loc(program, "uWave"));
    this.setSurfaceUniforms(program);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    this.dye.swap();
  }

  private render(): void {
    const gl = this.gl;
    const program = this.programs.display;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.useProgram(program);
    this.bind(0, this.dye.read.tex, this.loc(program, "uDye"));
    this.bind(1, this.waveTex, this.loc(program, "uWave"));
    this.setSurfaceUniforms(program);
    gl.uniform2f(this.loc(program, "uCanvas"), this.canvas.width, this.canvas.height);
    this.uploadActors(program);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  private uploadActors(program: WebGLProgram): void {
    const gl = this.gl;
    this.dropData.fill(0);
    this.dropColors.fill(0);
    this.dropMeta.fill(0);
    const width = Math.max(1, this.canvas.width);
    const count = Math.min(MAX_DROPS, this.drops.length);
    for (let i = 0; i < count; i++) {
      const drop = this.drops[i];
      const speed = Math.abs(drop.vy);
      const stretch = drop.state === "fall" ? 1 + Math.min(0.95, speed * 0.48) : 1;
      const radius = width * 0.052;
      const wobble = Math.sin(drop.age * 13 + drop.seed * 6) * 0.0035 * Math.min(1, drop.age * 5);
      this.dropData.set([drop.x, drop.y, radius, stretch], i * 4);
      this.dropColors.set(drop.rgb, i * 3);
      this.dropMeta.set([drop.squash, wobble, 0, 0], i * 4);
    }
    gl.uniform1i(this.loc(program, "uDropCount"), count);
    gl.uniform4fv(this.loc(program, "uDrops[0]"), this.dropData);
    gl.uniform3fv(this.loc(program, "uDropColor[0]"), this.dropColors);
    gl.uniform4fv(this.loc(program, "uDropMeta[0]"), this.dropMeta);

    this.sparkData.fill(0);
    this.sparkColors.fill(0);
    const sparks = Math.min(MAX_SPARKS, this.sparks.length);
    for (let i = 0; i < sparks; i++) {
      const spark = this.sparks[i];
      const alpha = 1 - spark.age / spark.life;
      this.sparkData.set([spark.x, spark.y, spark.radius * width, Math.max(0, alpha)], i * 4);
      this.sparkColors.set(spark.rgb, i * 3);
    }
    gl.uniform1i(this.loc(program, "uSparkCount"), sparks);
    gl.uniform4fv(this.loc(program, "uSparks[0]"), this.sparkData);
    gl.uniform3fv(this.loc(program, "uSparkColor[0]"), this.sparkColors);

    this.rippleData.fill(0);
    const ripples = Math.min(MAX_RIPPLES, this.ripples.length);
    for (let i = 0; i < ripples; i++) {
      const ripple = this.ripples[i];
      this.rippleData.set([ripple.x, ripple.age, ripple.strength, 0], i * 4);
    }
    gl.uniform1i(this.loc(program, "uRippleCount"), ripples);
    gl.uniform4fv(this.loc(program, "uRipples[0]"), this.rippleData);
  }

  private stepDrops(dt: number): void {
    for (let i = this.drops.length - 1; i >= 0; i--) {
      const drop = this.drops[i];
      drop.age += dt;
      if (drop.state === "fall") {
        drop.vy -= GRAVITY * dt;
        drop.y += drop.vy * dt;
        const surface = this.surfaceAt(drop.x);
        if (drop.y <= surface) {
          drop.y = surface;
          drop.vy = 0;
          drop.state = "squash";
          drop.squash = 0;
          this.impact(drop);
        }
      } else {
        drop.squash = Math.min(1, drop.squash + dt / 0.16);
        drop.y = this.surfaceAt(drop.x);
        if (drop.squash >= 1) {
          this.drops.splice(i, 1);
        }
      }
    }
    while (this.pendingDrops.length > 0 && this.drops.length < MAX_DROPS) {
      const next = this.pendingDrops.shift();
      if (next) this.pushDrop(next.x, next.rgb);
    }
  }

  private pushDrop(x: number, rgb: RGB): void {
    this.drops.push({
      x,
      y: 1.035,
      vy: -0.08,
      rgb,
      absorption: pigmentAbsorption(rgb),
      age: 0,
      squash: 0,
      state: "fall",
      seed: Math.random(),
    });
  }

  private impact(drop: Drop): void {
    const surface = this.surfaceAt(drop.x);
    const y = surface - 0.02;
    const a = drop.absorption;
    const j = (drop.seed - 0.5) * 0.012;
    const spin = drop.seed > 0.5 ? 1 : -1;

    this.dyeSplats.push(
      { x: drop.x, y, radius: 13, scale: 1.45, stretchX: 1.05, stretchY: 1.45, absorption: a },
      { x: drop.x, y: y - 0.04, radius: 9, scale: 0.9, stretchX: 0.7, stretchY: 2.4, absorption: a },
      { x: drop.x + j, y: y - 0.085, radius: 7, scale: 0.48, stretchX: 0.55, stretchY: 2.8, absorption: a },
      {
        x: drop.x + 0.022 + j * 0.3,
        y: y - 0.035,
        radius: 5.5,
        scale: 0.28,
        stretchX: 1.7,
        stretchY: 0.85,
        absorption: a,
      },
      {
        x: drop.x - 0.02,
        y: y - 0.055,
        radius: 4.5,
        scale: 0.22,
        stretchX: 1.2,
        stretchY: 1.8,
        absorption: a,
      },
    );

    this.vortices.push(
      {
        x: drop.x,
        y: y - 0.01,
        sep: 8 + drop.seed * 5,
        radius: 12 + drop.seed * 4,
        strength: 38 + drop.seed * 14,
        jet: 18 + drop.seed * 8,
        spin,
        spread: 16,
      },
      {
        x: drop.x + j,
        y: y - 0.05,
        sep: 5,
        radius: 8,
        strength: 18,
        jet: 8,
        spin: -spin,
        spread: 8,
      },
    );

    this.impulseWave(drop.x, 1.35);
    this.ripples.push({ x: drop.x, age: 0, strength: 1 });
    if (this.ripples.length > MAX_RIPPLES + 2) this.ripples.shift();

    for (let i = 0; i < 7; i++) {
      const spread = (i / 6 - 0.5) * 1.15;
      const upward = 0.16 + Math.random() * 0.28;
      this.sparks.push({
        x: drop.x + spread * 0.012,
        y: surface + 0.004,
        vx: spread * (0.05 + Math.random() * 0.04),
        vy: upward,
        age: 0,
        life: 0.32 + Math.random() * 0.38,
        rgb: drop.rgb,
        radius: 0.004 + Math.random() * 0.004,
      });
    }
    if (this.sparks.length > MAX_SPARKS) {
      this.sparks.splice(0, this.sparks.length - MAX_SPARKS);
    }
  }

  private stepSparks(dt: number): void {
    for (let i = this.sparks.length - 1; i >= 0; i--) {
      const spark = this.sparks[i];
      spark.age += dt;
      spark.vy -= 2.15 * dt;
      spark.x += spark.vx * dt;
      spark.y += spark.vy * dt;
      if (spark.age >= spark.life || spark.y < 0) {
        this.sparks.splice(i, 1);
      }
    }
    for (const ripple of this.ripples) ripple.age += dt;
    for (let i = this.ripples.length - 1; i >= 0; i--) {
      if (this.ripples[i].age > 2.6) this.ripples.splice(i, 1);
    }
  }

  private stepWaves(dt: number): void {
    const c2 = 760;
    const damp = 2.6;
    const n = WAVE_N;
    for (let i = 1; i < n - 1; i++) {
      const lap = this.h[i - 1] + this.h[i + 1] - 2 * this.h[i];
      this.v[i] += (c2 * lap - damp * this.v[i]) * dt;
    }
    this.v[0] = this.v[1];
    this.v[n - 1] = this.v[n - 2];
    for (let i = 0; i < n; i++) {
      this.h[i] += this.v[i] * dt;
      this.h[i] = clamp(this.h[i], -0.028, 0.028);
    }
    this.h[0] *= 0.9;
    this.h[n - 1] *= 0.9;
  }

  private impulseWave(x: number, strength: number): void {
    const n = WAVE_N;
    for (let i = 0; i < n; i++) {
      const dx = (i + 0.5) / n - x;
      const g = Math.exp(-(dx * dx) / (2 * 0.011 * 0.011));
      this.v[i] -= strength * g;
    }
  }

  private uploadWaves(): void {
    for (let i = 0; i < WAVE_N; i++) {
      const encoded = clamp(this.h[i] / WAVE_ENCODE + 0.5, 0, 1);
      const b = (encoded * 255) | 0;
      const o = i * 4;
      this.waveBytes[o] = b;
      this.waveBytes[o + 1] = b;
      this.waveBytes[o + 2] = b;
      this.waveBytes[o + 3] = 255;
    }
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.waveTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, WAVE_N, 1, gl.RGBA, gl.UNSIGNED_BYTE, this.waveBytes);
  }

  private surfaceAt(x: number): number {
    const n = WAVE_N;
    const fx = clamp(x, 0, 1) * (n - 1);
    const i = Math.floor(fx);
    const t = fx - i;
    const h0 = this.h[i] ?? 0;
    const h1 = this.h[Math.min(n - 1, i + 1)] ?? h0;
    const edge = Math.abs(x - 0.5) * 2;
    const meniscus = edge ** 3 * MENISCUS;
    return SURFACE + h0 * (1 - t) + h1 * t + meniscus + idleWave(x, this.time);
  }
}

function idleWave(x: number, time: number): number {
  return Math.sin(x * 18 + time * 1.1) * 0.0016 + Math.sin(x * 31 - time * 0.7) * 0.0008;
}

function clamp(v: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, v));
}

const programIds = new WeakMap<WebGLProgram, number>();
let nextProgramId = 1;
function programToId(program: WebGLProgram): number {
  let id = programIds.get(program);
  if (id === undefined) {
    id = nextProgramId++;
    programIds.set(program, id);
  }
  return id;
}
