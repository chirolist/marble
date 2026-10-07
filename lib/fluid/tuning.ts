/** Simulation grid. Matches the glass aspect ratio (5:8) so swirls stay round. */
export const SIM_W = 300;
export const SIM_H = 480;

/** Waterline as a fraction of the texture, y = 0 at the bottom. */
export const SURFACE = 0.785;

export const WAVE_N = 160;
/** Height encoding range of the 1D wave into an 8-bit texture. */
export const WAVE_ENCODE = 0.08;
export const MENISCUS = 0.014;

export const PRESSURE_ITERS = 22;

/** Velocity damping per second. High enough that the entry swirl settles. */
export const VEL_DECAY = 1.35;
/** Ink fading per second. Slow enough that color stays until reset. */
export const DYE_DECAY = 0.008;
/** Dye diffusion rate per second. Kept low so filaments are not a blur. */
export const DYE_DIFFUSE = 0.1;

/** Downward acceleration scale, in texels per second squared. */
export const BUOYANCY = 16;
/** Vorticity confinement. Keeps swirls alive without tearing the plume apart. */
export const CONFINEMENT = 1.1;
/** Ambient curl, multiplied by dt. Gentle marble-like drift where ink exists. */
export const AMBIENT = 5;

export const GRAVITY = 1.15;
export const MAX_DROPS = 8;
export const MAX_SPARKS = 16;
export const MAX_RIPPLES = 5;
export const MAX_VORTICES = 4;
export const MAX_SPLATS = 8;
