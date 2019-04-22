(function() {

const INITIALIZE_BUCKET_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

out ivec2 o_bucket;

uniform sampler2D u_positionTexture;
uniform float u_bucketSize;
uniform int u_particleNum;
uniform ivec2 u_bucketNum;

int convertCoordToIndex(ivec2 coord, int sizeX) {
  return coord.x + sizeX * coord.y;
}

int getBucketIndex(vec2 position) {
  ivec2 bucketCoord = ivec2(position / u_bucketSize);
  return bucketCoord.x + u_bucketNum.x * bucketCoord.y;
}

void main(void) {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  int particleTextureSizeX = textureSize(u_positionTexture, 0).x;
  int particleIndex = convertCoordToIndex(coord, particleTextureSizeX);
  if (particleIndex >= u_particleNum) {
    o_bucket = ivec2(65535, 65535);
    return;
  }
  vec2 position = texelFetch(u_positionTexture, coord, 0).xy;
  int bucketIndex = getBucketIndex(position);
  o_bucket = ivec2(bucketIndex, particleIndex);
}
`;

  const SWAP_BUCKET_INDEX_FRAGMENT_SHADER_SOURCE = 
`#version 300 es

precision highp float;
precision highp isampler2D;

out ivec2 o_bucket;

uniform isampler2D u_bucketTexture;
uniform int u_size;
uniform int u_blockStep;
uniform int u_subBlockStep;

int convertCoordToIndex(ivec2 coord, int sizeX) {
  return coord.x + sizeX * coord.y;
}

ivec2 convertIndexToCoord(int index, int sizeX) {
  return ivec2(index % sizeX, index / sizeX);
}

void main(void) {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  int index = convertCoordToIndex(coord, u_size);
  int d = 1 << (u_blockStep - u_subBlockStep);

  bool up = ((index >> u_blockStep) & 2) == 0;

  int targetIndex;
  bool first = (index & d) == 0;
  if (first) {
    targetIndex = index | d;
  } else {
    targetIndex = index & ~d;
    up = !up;
  }

  ivec2 a = texelFetch(u_bucketTexture, coord, 0).xy;
  ivec2 b = texelFetch(u_bucketTexture, ivec2(convertIndexToCoord(targetIndex, u_size)), 0).xy;

  if (a.x == b.x || (a.x >= b.x) == up) {
    o_bucket = b;
  } else {
    o_bucket = a;
  }
}

`;

  const INITIALIZE_BUCKET_REFERRER_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;
precision highp isampler2D;

out ivec2 o_referrer;

uniform ivec2 u_bucketReferrerTextureSize;
uniform isampler2D u_bucketTexture;
uniform int u_particleTextureSizeN;
uniform ivec2 u_bucketNum;

int convertCoordToIndex(ivec2 coord, int sizeX) {
  return coord.x + sizeX * coord.y;
}

ivec2 convertIndexToCoord(int index, int sizeX) {
  return ivec2(index % sizeX, index / sizeX);
}

int getBucketIndex(int particleIndex, int particleTextureSizeX) {
  return texelFetch(u_bucketTexture, convertIndexToCoord(particleIndex, particleTextureSizeX), 0).x;
}

int binarySearchMinIndex(int target, int from, int to, int particleTextureSizeX) {
  for (int i = 0; i < u_particleTextureSizeN + 1; i++) {
    int middle = from + (to - from) / 2;
    int bucketIndex = getBucketIndex(middle, particleTextureSizeX);
    if (bucketIndex < target) {
      from = middle + 1;
    } else {
      to = middle;
    }
    if (from == to) {
      if (getBucketIndex(from, particleTextureSizeX) == target) {
        return from;
      } else {
        return -1;
      }
    }
  }
  return -1;
}

int binarySearchMaxIndex(int target, int from, int to, int particleTextureSizeX) {
  for (int i = 0; i < u_particleTextureSizeN + 1; i++) {
    int middle = from + (to - from) / 2 + 1;
    int bucketIndex = getBucketIndex(middle, particleTextureSizeX);
    if (bucketIndex > target) {
      to = middle - 1;
    } else {
      from = middle;
    }
    if (from == to) {
      if (getBucketIndex(from, particleTextureSizeX) == target) {
        return from;
      } else {
        return -1;
      }
    }
  }
  return -1;
}

ivec2 binarySearchRange(int target, int from, int to, int particleTextureSizeX) {
  from = binarySearchMinIndex(target, from, to, particleTextureSizeX);
  to = from == -1 ? -1 : binarySearchMaxIndex(target, from, to, particleTextureSizeX);
  return ivec2(from, to);
}

void main(void) {
  int bucketIndex = convertCoordToIndex(ivec2(gl_FragCoord.xy), u_bucketReferrerTextureSize.x);
  int maxBucketIndex = u_bucketNum.x * u_bucketNum.y;

  if (bucketIndex >= maxBucketIndex) {
    o_referrer = ivec2(-1, -1);
    return;
  }

  ivec2 particleTextureSize = textureSize(u_bucketTexture, 0);
  o_referrer = binarySearchRange(bucketIndex, 0, particleTextureSize.x * particleTextureSize.y - 1, particleTextureSize.x);
}
`;

const INITIALIZE_PARTICLE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

layout (location = 0) out vec2 o_position;
layout (location = 1) out vec2 o_velocity;

uniform int u_particleNum;
uniform int u_particleTextureSizeX;
uniform float u_targetSpace;
uniform int u_damParticleNumX;

int convertCoordToIndex(ivec2 coord, int sizeX) {
  return coord.x + sizeX * coord.y;
}

void main(void) {
  int index = convertCoordToIndex(ivec2(gl_FragCoord.xy), u_particleTextureSizeX);

  if (index >= u_particleNum) { // unused pixels
    o_position = vec2(0.0);
    o_velocity = vec2(0.0);
    return;
  }

  ivec2 posIndex= ivec2(index % u_damParticleNumX, index / u_damParticleNumX);

  float halfTargetSpace = 0.5 * u_targetSpace;

  o_position = vec2(
    float(posIndex.x) * u_targetSpace,
    float(posIndex.y) * 0.5 * sqrt(3.0) * u_targetSpace
  ) + vec2(posIndex.y % 2 == 0 ? halfTargetSpace : u_targetSpace, halfTargetSpace);
  o_velocity = vec2(0.0);
}
`;

  const COMPUTE_DENSITY_AND_PRESSURE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;
precision highp isampler2D;

#define PI 3.14159265359

layout (location = 0) out float o_density;
layout (location = 1) out float o_pressure;

uniform isampler2D u_bucketTexture;
uniform isampler2D u_bucketReferrerTexture;
uniform int u_particleNum;
uniform ivec2 u_bucketNum;

uniform sampler2D u_positionTexture;
uniform float u_kernelRadius;
uniform float u_mass;
uniform float u_targetDensity;
uniform float u_speedOfSound;
uniform float u_eosExponent;
uniform float u_negativePressureScale;

float StdKernelCoef;

int convertCoordToIndex(ivec2 coord, int sizeX) {
  return coord.x + sizeX * coord.y;
}

ivec2 convertIndexToCoord(int index, int sizeX) {
  return ivec2(index % sizeX, index / sizeX);
}

void initializeConstants() {
  StdKernelCoef = 4.0 / (PI * u_kernelRadius * u_kernelRadius * u_kernelRadius * u_kernelRadius * u_kernelRadius * u_kernelRadius * u_kernelRadius * u_kernelRadius);
}

float calcStdKernel(float d) {
  if (d < u_kernelRadius) {
    float x = u_kernelRadius * u_kernelRadius - d * d;
    return StdKernelCoef * x * x * x;
  }
  return 0.0;
}

float computeDensityFromBucket(int particleIndex, vec2 position, ivec2 bucketPosition, ivec2 bucketNum, int particleTextureSizeX, int bucketReferrerTextureSizeX) {
  if (bucketPosition.x < 0 || bucketPosition.x >= bucketNum.x ||
      bucketPosition.y < 0 || bucketPosition.y >= bucketNum.y) {
    return 0.0;
  }
  int bucketIndex = bucketPosition.x + bucketNum.x * bucketPosition.y;
  ivec2 bucketReferrer = texelFetch(u_bucketReferrerTexture, convertIndexToCoord(bucketIndex, bucketReferrerTextureSizeX), 0).xy;

  if (bucketReferrer.x == -1 || bucketReferrer.y == -1) {
    return 0.0;
  }

  float density = 0.0;
  for (int i = bucketReferrer.x; i <= bucketReferrer.y; i++) {
    int otherIndex = texelFetch(u_bucketTexture, convertIndexToCoord(i, particleTextureSizeX), 0).y;
    if (particleIndex == otherIndex) {
      continue;
    }
    vec2 otherPos = texelFetch(u_positionTexture, convertIndexToCoord(otherIndex, particleTextureSizeX), 0).xy;
    float dist = length(position - otherPos);
    density += calcStdKernel(dist);
  }
  return density;
}

float computeDensity(int particleIndex, vec2 position, int particleTextureSizeX) {
  int bucketReferrerTextureSizeX = textureSize(u_bucketReferrerTexture, 0).x;

  vec2 bucketPosition = position / (2.0 * u_kernelRadius);
  int xOffset = fract(bucketPosition.x) < 0.5 ? -1 : 1;
  int yOffset = fract(bucketPosition.y) < 0.5 ? -1 : 1;

  ivec2 bucketPosition00 = ivec2(bucketPosition);
  ivec2 bucketPosition10 = bucketPosition00 + ivec2(xOffset, 0);
  ivec2 bucketPosition01 = bucketPosition00 + ivec2(0, yOffset);
  ivec2 bucketPosition11 = bucketPosition00 + ivec2(xOffset, yOffset);

  float density = 0.0;
  density += computeDensityFromBucket(particleIndex, position, bucketPosition00, u_bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX);
  density += computeDensityFromBucket(particleIndex, position, bucketPosition10, u_bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX);
  density += computeDensityFromBucket(particleIndex, position, bucketPosition01, u_bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX);
  density += computeDensityFromBucket(particleIndex, position, bucketPosition11, u_bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX);

  return u_mass * density;
}

float computePressure(float density) {
  float eosScale = u_targetDensity * u_speedOfSound / u_eosExponent;
  float pressure = eosScale / u_eosExponent * (pow(density / u_targetDensity, u_eosExponent) - 1.0);
  if (pressure < 0.0) {
    pressure *= u_negativePressureScale;
  }
  return pressure;
}

void main(void) {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  int particleTextureSizeX = textureSize(u_positionTexture, 0).x;
  int particleIndex = convertCoordToIndex(coord, particleTextureSizeX);
  if (particleIndex >= u_particleNum) {
    o_density = 0.0;
    o_pressure = 0.0;
    return;
  }
  initializeConstants();

  vec2 position = texelFetch(u_positionTexture, coord, 0).xy;
  float density = computeDensity(particleIndex, position, particleTextureSizeX);
  float pressure = computePressure(density);
  o_density = density;
  o_pressure = pressure;
}
`;

  const UPDATE_PARTICLE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;
precision highp isampler2D;

#define PI 3.14159265359

layout (location = 0) out vec2 o_position;
layout (location = 1) out vec2 o_velocity;

uniform isampler2D u_bucketTexture;
uniform isampler2D u_bucketReferrerTexture;
uniform int u_particleNum;
uniform ivec2 u_bucketNum;

uniform sampler2D u_positionTexture;
uniform sampler2D u_velocityTexture;
uniform sampler2D u_densityTexture;
uniform sampler2D u_pressureTexture;
uniform vec2 u_simulationSpace;
uniform float u_targetSpace;
uniform float u_kernelRadius;
uniform float u_mass;
uniform vec2 u_gravity;
uniform float u_viscosityCoef;
uniform float u_restitutionCoef;
uniform float u_frictionCoef;
uniform float u_deltaTime;

int convertCoordToIndex(ivec2 coord, int sizeX) {
  return coord.x + sizeX * coord.y;
}

ivec2 convertIndexToCoord(int index, int sizeX) {
  return ivec2(index % sizeX, index / sizeX);
}

float SpikyKernelDerivativeCoef, ViscosityKernelSecondDerivativeCoef;
void initializeConstants() {
  SpikyKernelDerivativeCoef = -30.0 / (PI * u_kernelRadius * u_kernelRadius * u_kernelRadius * u_kernelRadius * u_kernelRadius);
  ViscosityKernelSecondDerivativeCoef = 20.0 / (3.0 * PI * u_kernelRadius * u_kernelRadius * u_kernelRadius * u_kernelRadius * u_kernelRadius);
}

float calcSpikyKernelDerivative(float d) {
  if (d < u_kernelRadius) {
    float x = u_kernelRadius - d;
    return SpikyKernelDerivativeCoef * x * x;
  }
  return 0.0;
}

vec2 calcSpikyKernelGradient(vec2 center, vec2 position) {
  vec2 v = center - position;
  if (length(v) == 0.0) {
    return vec2(0.0);
  }
  return calcSpikyKernelDerivative(length(v)) * normalize(v);
}

float calcViscosityKernelSecondDerivative(float d) {
  if (d < u_kernelRadius) {
    return ViscosityKernelSecondDerivativeCoef * (u_kernelRadius - d);
  }
  return 0.0;
}

void accumulatePressureAndViscosityForces(int particleIndex, vec2 position, vec2 velocity, float density, float pressure, ivec2 bucketPosition, ivec2 bucketNum, int particleTextureSizeX, int bucketReferrerTextureSizeX, inout vec2 pressureForce, inout vec2 viscosityForce) {
  if (bucketPosition.x < 0 || bucketPosition.x >= bucketNum.x ||
      bucketPosition.y < 0 || bucketPosition.y >= bucketNum.y) {
    return;
  }
  int bucketIndex = bucketPosition.x + bucketNum.x * bucketPosition.y;
  ivec2 coord = ivec2(convertIndexToCoord(bucketIndex, bucketReferrerTextureSizeX));

  ivec2 bucketReferrer = texelFetch(u_bucketReferrerTexture, coord, 0).xy;
  if (bucketReferrer.x == -1 || bucketReferrer.y == -1) {
    return;
  }

  float density2 = density * density;
  for (int i = bucketReferrer.x; i <= bucketReferrer.y; i++) {
    int otherIndex = texelFetch(u_bucketTexture, convertIndexToCoord(i, particleTextureSizeX), 0).y;
    if (particleIndex == otherIndex) {
      continue;
    }
    ivec2 otherCoord = convertIndexToCoord(otherIndex, particleTextureSizeX);
    vec2 otherPos = texelFetch(u_positionTexture, otherCoord, 0).xy;

    float dist = length(position - otherPos);

    if (density != 0.0 && dist < u_kernelRadius) {
      float otherDens = texelFetch(u_densityTexture, otherCoord, 0).x;
      if (otherDens == 0.0) { continue; }
      vec2 otherVel = texelFetch(u_velocityTexture, otherCoord, 0).xy;
      float otherPres = texelFetch(u_pressureTexture, otherCoord, 0).x;

      float otherDens2 = otherDens * otherDens;
      pressureForce += -(pressure / density2 + otherPres / otherDens2) * calcSpikyKernelGradient(position, otherPos);

      viscosityForce += (otherVel - velocity) * calcViscosityKernelSecondDerivative(dist) / otherDens;
    }
  }
}

vec2 computePressureAndViscosityForces(int particleIndex, vec2 position, vec2 velocity, float density, float pressure, int particleTextureSizeX) {
  int bucketReferrerTextureSizeX = textureSize(u_bucketReferrerTexture, 0).x;

  vec2 bucketPosition = position / (2.0 * u_kernelRadius);
  int xOffset = fract(bucketPosition.x) < 0.5 ? -1 : 1;
  int yOffset = fract(bucketPosition.y) < 0.5 ? -1 : 1;

  ivec2 bucketPosition00 = ivec2(bucketPosition);
  ivec2 bucketPosition10 = bucketPosition00 + ivec2(xOffset, 0);
  ivec2 bucketPosition01 = bucketPosition00 + ivec2(0, yOffset);
  ivec2 bucketPosition11 = bucketPosition00 + ivec2(xOffset, yOffset);

  vec2 pressureForce = vec2(0.0);
  vec2 viscosityForce = vec2(0.0);
  accumulatePressureAndViscosityForces(particleIndex, position, velocity, density, pressure, bucketPosition00, u_bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX, pressureForce, viscosityForce);
  accumulatePressureAndViscosityForces(particleIndex, position, velocity, density, pressure, bucketPosition10, u_bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX, pressureForce, viscosityForce);
  accumulatePressureAndViscosityForces(particleIndex, position, velocity, density, pressure, bucketPosition01, u_bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX, pressureForce, viscosityForce);
  accumulatePressureAndViscosityForces(particleIndex, position, velocity, density, pressure, bucketPosition11, u_bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX, pressureForce, viscosityForce);
  pressureForce *= u_mass;
  viscosityForce *= u_mass * u_mass * u_viscosityCoef;

  return pressureForce + viscosityForce;
}

vec2 computeExternalForce() {
  return u_mass * u_gravity;
}

vec2 computeForce(int particleIndex, vec2 position, vec2 velocity, float density, float pressure, int particleTextureSizeX) {
  vec2 force = vec2(0.0);
  force += density == 0.0 ? vec2(0.0) : computePressureAndViscosityForces(particleIndex, position, velocity, density, pressure, particleTextureSizeX);
  force += computeExternalForce();
  return force;
}

void collideWall(vec2 center, vec2 normal, inout vec2 position, inout vec2 velocity) {
  vec2 toP = position - center;
  float dotToP = dot(normal, toP);
  vec2 toPN = normal * dotToP;
  vec2 toPT = toP - toPN;
  float dist = length(toPN);
  if (dotToP >= 0.0 && dist > 0.5 * u_targetSpace) {
    return;
  }
  vec2 nearestPos = center + toPT;
  position = nearestPos + 0.5 * u_targetSpace * normal;
  if (dot(velocity, normal) < 0.0) {
    vec2 velN = u_restitutionCoef * (normal * dot(normal, velocity));
    vec2 velT = u_frictionCoef * (velocity - velN);
    velocity = -velN + velT;
  }
}

void solveCollision(inout vec2 position, inout vec2 velocity) {
  vec2 halfSimSpace = 0.5 * u_simulationSpace;
  collideWall(vec2(halfSimSpace.x, 0.0), vec2(0.0, 1.0), position, velocity);
  collideWall(vec2(0.0, halfSimSpace.y), vec2(1.0, 0.0), position, velocity);
  collideWall(vec2(halfSimSpace.x, u_simulationSpace.y), vec2(0.0, -1.0), position, velocity);
  collideWall(vec2(u_simulationSpace.x, halfSimSpace.y), vec2(-1.0, 0.0), position, velocity);
}

void main(void) {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  int particleTextureSizeX = textureSize(u_positionTexture, 0).x;
  int particleIndex = convertCoordToIndex(coord, particleTextureSizeX);
  if (particleIndex >= u_particleNum) {
    o_position = vec2(0.0);
    o_velocity = vec2(0.0);
    return;
  }

  initializeConstants();

  vec2 position = texelFetch(u_positionTexture, coord, 0).xy;
  vec2 velocity = texelFetch(u_velocityTexture, coord, 0).xy;
  float density = texelFetch(u_densityTexture, coord, 0).x;
  float pressure = texelFetch(u_pressureTexture, coord, 0).x;

  vec2 force = computeForce(particleIndex, position, velocity, density, pressure, particleTextureSizeX);
  vec2 nextVelocity = velocity + u_deltaTime * force;
  vec2 nextPosition = position + u_deltaTime * nextVelocity;
  solveCollision(nextPosition, nextVelocity);

  o_position = nextPosition;
  o_velocity = nextVelocity;
}
`;

  const RENDER_PARTICLE_VERTEX_SHADER_SOURCE = 
`#version 300 es

out vec3 v_color;

uniform sampler2D u_positionTexture;
uniform sampler2D u_velocityTexture;
uniform vec2 u_viewportSize;
uniform vec2 u_simulationSpace;
uniform float u_targetSpace;
uniform float u_particleSize;

ivec2 convertIndexToCoord(int index, int sizeX) {
  return ivec2(index % sizeX, index / sizeX);
}

void main(void) {
  ivec2 coord = convertIndexToCoord(gl_VertexID, textureSize(u_positionTexture, 0).x);
  vec2 position = texelFetch(u_positionTexture, coord, 0).xy;
  vec2 viewportScale = min(u_viewportSize.x, u_viewportSize.y) / u_viewportSize;
  float minSimSpace = min(u_simulationSpace.x, u_simulationSpace.y);
  vec2 clipPos = viewportScale * ((position / minSimSpace) * 2.0 - u_simulationSpace / minSimSpace);
  gl_Position = vec4(clipPos, 0.0, 1.0);
  gl_PointSize = u_particleSize;
  vec2 velocity = texelFetch(u_velocityTexture, coord, 0).xy;
  v_color = mix(vec3(0.1, 1.0, 0.7), vec3(1.0, 0.5, 0.2), smoothstep(0.0, 0.5, length(velocity)));
}
`;

  const RENDER_PARTICLE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

in vec3 v_color;

out vec4 o_color;

void main(void) {
  if (length(gl_PointCoord - 0.5) > 0.5) {
    discard;
  } else {
    o_color = vec4(v_color, 1.0);
  }
}
`;

  const createParticleFramebuffer = function(gl, size) {
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    const positionTexture = createTexture(gl, size, size, gl.RG32F, gl.RG, gl.FLOAT);
    gl.bindTexture(gl.TEXTURE_2D, positionTexture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, positionTexture, 0);
    const velocityTexture = createTexture(gl, size, size, gl.RG32F, gl.RG, gl.FLOAT);
    gl.bindTexture(gl.TEXTURE_2D, velocityTexture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, velocityTexture, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return {
      framebuffer: framebuffer,
      positionTexture: positionTexture,
      velocityTexture: velocityTexture
    };
  };

  const createDensityAndPressureFramebuffer = function(gl, size) {
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    const densityTexture = createTexture(gl, size, size, gl.R32F, gl.RED, gl.FLOAT);
    gl.bindTexture(gl.TEXTURE_2D, densityTexture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, densityTexture, 0);
    const pressureTexture = createTexture(gl, size, size, gl.R32F, gl.RED, gl.FLOAT);
    gl.bindTexture(gl.TEXTURE_2D, pressureTexture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, pressureTexture, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return {
      framebuffer: framebuffer,
      densityTexture: densityTexture,
      pressureTexture: pressureTexture
    };
  };

  function createBucketFramebuffer(gl, size) {
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    const bucketTexture = createTexture(gl, size, size, gl.RG32I, gl.RG_INTEGER, gl.INT);
    gl.bindTexture(gl.TEXTURE_2D, bucketTexture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, bucketTexture, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return {
      framebuffer: framebuffer,
      bucketTexture: bucketTexture
    };
  }

  function createBucketReferrerFramebuffer(gl, size) {
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    const bucketReferrerTexture = createTexture(gl, size, size, gl.RG32I, gl.RG_INTEGER, gl.INT);
    gl.bindTexture(gl.TEXTURE_2D, bucketReferrerTexture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, bucketReferrerTexture, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return {
      framebuffer: framebuffer,
      bucketReferrerTexture: bucketReferrerTexture
    };
  }

  class StdKernel {
    constructor(kernelRadius) {
      this._h = kernelRadius;
      this._h2 = this._h * this._h;
      const h4 = this._h2 * this._h2;
      const h8 = h4 * h4;;
      this._coef = 4.0 / (Math.PI * h8);
    }
    value(d) {
      if (d < this._h) {
        const x = this._h2 - d * d;
        return this._coef * x * x * x;
      }
      return 0.0;
    }
  }

  function calcMass(kernelRadius, targetSpace, targetDensity) {
    const kernel = new StdKernel(kernelRadius);
    const yStep = 0.5 * Math.sqrt(3.0) * targetSpace;
    let useXOffset = false;
    let sum = 0.0;
    for (let y = 0.0; y < kernelRadius; y += yStep) {
      xOffset = useXOffset ? 0.5 * targetSpace: 0.0;
      for (let x = xOffset; x < kernelRadius; x += targetSpace) {
        if (x !== 0.0 && y === 0.0) {
          sum += 2.0 * kernel.value(new Vector2(x, y).mag());
        } else if (x == 0.0 && y !== 0.0) {
          sum += kernel.value(new Vector2(x, y).mag());
        } else if (x !== 0.0 && y !== 0.0) {
          sum += 4.0 * kernel.value(new Vector2(x, y).mag());
        }
      }
      useXOffset = !useXOffset;
    }
    return targetDensity / sum;
  }

  const stats = new Stats();
  document.body.appendChild(stats.dom);

  const parameters = {
    dynamic: {
      'viscosity coef': 0.5,
      'restitution coef': 1.0,
      'friction coef': 1.0,
      'gravity': 9.8,
      'time step': 0.0005,
      'time scale': 0.3,
      'particle size': 5.0,
    },
    static: {
      'dam width': 0.8,
      'dam height': 0.7,
      'target space': 0.01,
      'kernel scale': 2.1,
    },
    'reset': _ => reset()
  };

  const gui = new dat.GUI();
  const dynamicFolder = gui.addFolder('dynamic parameters');
  dynamicFolder.add(parameters.dynamic, 'viscosity coef', 0.0, 2.0).step(0.01);
  dynamicFolder.add(parameters.dynamic, 'restitution coef', 0.0, 1.0).step(0.01);
  dynamicFolder.add(parameters.dynamic, 'friction coef', 0.0, 1.0).step(0.01);
  dynamicFolder.add(parameters.dynamic, 'gravity', 0.0, 19.6).step(0.01);
  dynamicFolder.add(parameters.dynamic, 'time step', 0.0001, 0.001).step(0.0001);
  dynamicFolder.add(parameters.dynamic, 'time scale', 0.0, 1.0).step(0.01);
  dynamicFolder.add(parameters.dynamic, 'particle size', 1.0, 20.0);
  const staticFolder =gui.addFolder('static parameters');
  staticFolder.add(parameters.static, 'dam width', 0.3, 1.0).step(0.01);
  staticFolder.add(parameters.static, 'dam height', 0.3, 1.0).step(0.01);
  staticFolder.add(parameters.static, 'target space', 0.005, 0.02).step(0.0001);
  staticFolder.add(parameters.static, 'kernel scale', 1.5, 3.5).step(0.01);
  gui.add(parameters, 'reset');

  const canvas = document.getElementById('canvas');
  const resizeCanvas = function() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  };
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  const gl = canvas.getContext('webgl2');
  gl.getExtension('EXT_color_buffer_float');
  gl.clearColor(0.0, 0.0, 0.05, 1.0);

  const initializeBucketProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, INITIALIZE_BUCKET_FRAGMENT_SHADER_SOURCE);
  const swapBucketIndexProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, SWAP_BUCKET_INDEX_FRAGMENT_SHADER_SOURCE);
  const initializeBucketReferrerProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, INITIALIZE_BUCKET_REFERRER_FRAGMENT_SHADER_SOURCE);
  const initializeParticleProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, INITIALIZE_PARTICLE_FRAGMENT_SHADER_SOURCE);
  const computeDensityAndPressureProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, COMPUTE_DENSITY_AND_PRESSURE_FRAGMENT_SHADER_SOURCE);
  const updateParticleProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, UPDATE_PARTICLE_FRAGMENT_SHADER_SOURCE);
  const renderParticleProgram = createProgramFromSource(gl, RENDER_PARTICLE_VERTEX_SHADER_SOURCE, RENDER_PARTICLE_FRAGMENT_SHADER_SOURCE);

  const initializeBucketUniforms = getUniformLocations(gl, initializeBucketProgram, [
    'u_positionTexture', 'u_bucketSize', 'u_particleNum', 'u_bucketNum'
  ]);
  const swapBucketIndexUniforms = getUniformLocations(gl, swapBucketIndexProgram, [
    'u_bucketTexture', 'u_size', 'u_blockStep', 'u_subBlockStep'
  ]);
  const initializeBucketReferrerUniforms = getUniformLocations(gl, initializeBucketReferrerProgram, [
    'u_bucketReferrerTextureSize', 'u_bucketTexture', 'u_particleTextureSizeN', 'u_bucketNum'
  ]);
  const initializeParticleUniforms = getUniformLocations(gl, initializeParticleProgram,[
    'u_particleNum', 'u_particleTextureSizeX', 'u_targetSpace', 'u_damParticleNumX'
  ]);
  const computeDensityAndPressureUniforms = getUniformLocations(gl, computeDensityAndPressureProgram,[
    'u_bucketTexture', 'u_bucketReferrerTexture', 'u_particleNum', 'u_bucketNum',
    'u_positionTexture', 'u_kernelRadius', 'u_mass', 'u_targetDensity', 'u_speedOfSound', 'u_eosExponent', 'u_negativePressureScale'
  ]);
  const updateParticleUniforms = getUniformLocations(gl, updateParticleProgram,[
    'u_bucketTexture', 'u_bucketReferrerTexture', 'u_particleNum', 'u_bucketNum',
    'u_positionTexture', 'u_velocityTexture', 'u_densityTexture', 'u_pressureTexture',
    'u_simulationSpace', 'u_targetSpace', 'u_kernelRadius', 'u_mass', 'u_gravity', 'u_viscosityCoef', 'u_restitutionCoef', 'u_frictionCoef', 'u_deltaTime'
  ]);
  const renderParticleUniforms = getUniformLocations(gl, renderParticleProgram, [
    'u_positionTexture', 'u_velocityTexture', 'u_viewportSize', 'u_simulationSpace', 'u_targetSpace', 'u_particleSize'
  ]);

  const fillViewportVao = createFillViewportVao(gl);

  let requestId = null;
  const reset = () => {
    if (requestId !== null) {
      cancelAnimationFrame(requestId);
    }

    const damWidth = parameters.static['dam width'];
    const damHeight = parameters.static['dam height'];
    const targetSpace = parameters.static['target space'];
    const simulationSpace = new Vector2(1.6, 1.2);
    const kernelRadius = targetSpace * parameters.static['kernel scale'];
    const bucketSize = 2.0 * kernelRadius;
    const targetDensity = 1000.0;
    const mass = calcMass(kernelRadius, targetSpace, targetDensity);
    const speedOfSound = 1481.0; // (m/s in 20 degree)
    const eosExponent = 7.0;
    const negativePressureScale = 0.0;
    const bucketNum = Vector2.div(simulationSpace, bucketSize).ceil().add(new Vector2(1, 1));
    const totalBuckets = bucketNum.x * bucketNum.y;

    const halfTargetSpace = 0.5 * targetSpace;

    const damParticleNum = new Vector2(
      Math.floor((damWidth - halfTargetSpace) / targetSpace),
      Math.floor((damHeight - halfTargetSpace) / (0.5 * Math.sqrt(3.0) * targetSpace))
    );
    const particleNum = damParticleNum.x * damParticleNum.y;
    if (particleNum > 65535) {
      throw new Error('number of particles is too large: ' + particleNum);
    }
    if (bucketNum > 65535) {
      throw new Error('number of buckets is too large: ' + bucketNum);
    }

    let particleTextureSize, particleTextureSizeN;
    for (particleTextureSizeN = 0; ; particleTextureSizeN++) {
      particleTextureSize = 2 ** particleTextureSizeN;
      if (particleNum < particleTextureSize * particleTextureSize) {
        break;
      }
    }

    let bucketReferrerTextureSize;
    for (let i = 0; ; i++) {
      bucketReferrerTextureSize = 2 ** i;
      if (totalBuckets < bucketReferrerTextureSize * bucketReferrerTextureSize) {
        break;
      }
    }

    let particleFbObjR = createParticleFramebuffer(gl, particleTextureSize);
    let particleFbObjW = createParticleFramebuffer(gl, particleTextureSize);
    const swapParticleFbObj = () => {
      const tmp = particleFbObjR;
      particleFbObjR = particleFbObjW;
      particleFbObjW = tmp;
    };
    const densityAndPressureFbObj = createDensityAndPressureFramebuffer(gl, particleTextureSize);
  
    let bucketFbObjR = createBucketFramebuffer(gl, particleTextureSize);;
    let bucketFbObjW = createBucketFramebuffer(gl, particleTextureSize);
    const swapBucketFbObj = () => {
      const tmp = bucketFbObjR;
      bucketFbObjR = bucketFbObjW;
      bucketFbObjW = tmp;
    };
    const bucketReferrerFbObj = createBucketReferrerFramebuffer(gl, bucketReferrerTextureSize);

    const initializeParticles = () => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, particleFbObjW.framebuffer);
      gl.viewport(0.0, 0.0, particleTextureSize, particleTextureSize);
      gl.useProgram(initializeParticleProgram);
      gl.uniform1i(initializeParticleUniforms['u_particleNum'], particleNum);
      gl.uniform1i(initializeParticleUniforms['u_particleTextureSizeX'], particleTextureSize);
      gl.uniform1f(initializeParticleUniforms['u_targetSpace'], targetSpace);
      gl.uniform1i(initializeParticleUniforms['u_damParticleNumX'], damParticleNum.x);
      gl.bindVertexArray(fillViewportVao);
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      swapParticleFbObj();
    };

    const initializeBucket = () => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, bucketFbObjW.framebuffer);
      gl.viewport(0.0, 0.0, particleTextureSize, particleTextureSize);
      gl.useProgram(initializeBucketProgram);
      setUniformTexture(gl, 0, particleFbObjR.positionTexture, initializeBucketUniforms['u_positionTexture']);
      gl.uniform1f(initializeBucketUniforms['u_bucketSize'], bucketSize);
      gl.uniform1i(initializeBucketUniforms['u_particleNum'], particleNum);
      gl.uniform2i(initializeBucketUniforms['u_bucketNum'], bucketNum.x, bucketNum.y);
      gl.bindVertexArray(fillViewportVao);
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      swapBucketFbObj();
    };

    const swapBucketIndex = (i, j) => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, bucketFbObjW.framebuffer);
      gl.viewport(0.0, 0.0, particleTextureSize, particleTextureSize);
      gl.useProgram(swapBucketIndexProgram);
      setUniformTexture(gl, 0, bucketFbObjR.bucketTexture, swapBucketIndexUniforms['u_bucketTexture']);
      gl.uniform1i(swapBucketIndexUniforms['u_size'], particleTextureSize, particleTextureSize);
      gl.uniform1i(swapBucketIndexUniforms['u_blockStep'], i);
      gl.uniform1i(swapBucketIndexUniforms['u_subBlockStep'], j);
      gl.bindVertexArray(fillViewportVao);
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      swapBucketFbObj();
    };

    const initializeBucketReferrer = () => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, bucketReferrerFbObj.framebuffer);
      gl.viewport(0.0, 0.0, bucketReferrerTextureSize, bucketReferrerTextureSize);
      gl.useProgram(initializeBucketReferrerProgram);
      setUniformTexture(gl, 0, bucketFbObjR.bucketTexture, initializeBucketReferrerUniforms['u_bucketTexture']);
      gl.uniform1i(initializeBucketReferrerUniforms['u_particleTextureSizeN'], 2 * particleTextureSizeN);
      gl.uniform2i(initializeBucketReferrerUniforms['u_bucketReferrerTextureSize'], bucketReferrerTextureSize, bucketReferrerTextureSize);
      gl.uniform2i(initializeBucketReferrerUniforms['u_bucketNum'], bucketNum.x, bucketNum.y);
      gl.bindVertexArray(fillViewportVao);
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    };

    const constructBuckets = () => {
      initializeBucket();
      for (let i = 0; i < 2 * particleTextureSizeN; i++) {
        for (let j = 0; j <= i; j++) {
          swapBucketIndex(i, j);
        }
      }
      initializeBucketReferrer();
    };

    const computeDensityAndPressure = () => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, densityAndPressureFbObj.framebuffer);
      gl.viewport(0.0, 0.0, particleTextureSize, particleTextureSize);
      gl.useProgram(computeDensityAndPressureProgram);
      setUniformTexture(gl, 0, bucketFbObjR.bucketTexture, computeDensityAndPressureUniforms['u_bucketTexture']);
      setUniformTexture(gl, 1, bucketReferrerFbObj.bucketReferrerTexture, computeDensityAndPressureUniforms['u_bucketReferrerTexture']);
      gl.uniform1i(computeDensityAndPressureUniforms['u_particleNum'], particleNum);
      gl.uniform2i(computeDensityAndPressureUniforms['u_bucketNum'], bucketNum.x, bucketNum.y);
      setUniformTexture(gl, 2, particleFbObjR.positionTexture, computeDensityAndPressureUniforms['u_positionTexture']);
      gl.uniform1f(computeDensityAndPressureUniforms['u_kernelRadius'], kernelRadius);
      gl.uniform1f(computeDensityAndPressureUniforms['u_mass'], mass);
      gl.uniform1f(computeDensityAndPressureUniforms['u_targetDensity'], targetDensity);
      gl.uniform1f(computeDensityAndPressureUniforms['u_speedOfSound'], speedOfSound);
      gl.uniform1f(computeDensityAndPressureUniforms['u_eosExponent'], eosExponent);
      gl.uniform1f(computeDensityAndPressureUniforms['u_negativePressureScale'], negativePressureScale);
      gl.bindVertexArray(fillViewportVao);
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    };

    const updateParticles = (deltaTime) => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, particleFbObjW.framebuffer);
      gl.viewport(0.0, 0.0, particleTextureSize, particleTextureSize);
      gl.useProgram(updateParticleProgram);
      setUniformTexture(gl, 0, bucketFbObjR.bucketTexture, updateParticleUniforms['u_bucketTexture']);
      setUniformTexture(gl, 1, bucketReferrerFbObj.bucketReferrerTexture, updateParticleUniforms['u_bucketReferrerTexture']);
      gl.uniform1i(updateParticleUniforms['u_particleNum'], particleNum);
      gl.uniform2i(updateParticleUniforms['u_bucketNum'], bucketNum.x, bucketNum.y);
      setUniformTexture(gl, 2, particleFbObjR.positionTexture, updateParticleUniforms['u_positionTexture']);
      setUniformTexture(gl, 3, particleFbObjR.velocityTexture, updateParticleUniforms['u_velocityTexture']);
      setUniformTexture(gl, 4, densityAndPressureFbObj.densityTexture, updateParticleUniforms['u_densityTexture']);
      setUniformTexture(gl, 5, densityAndPressureFbObj.pressureTexture, updateParticleUniforms['u_pressureTexture']);
      gl.uniform2f(updateParticleUniforms['u_simulationSpace'], simulationSpace.x, simulationSpace.y);
      gl.uniform1f(updateParticleUniforms['u_targetSpace'], targetSpace);
      gl.uniform1f(updateParticleUniforms['u_kernelRadius'], kernelRadius);
      gl.uniform1f(updateParticleUniforms['u_mass'], mass);
      gl.uniform2f(updateParticleUniforms['u_gravity'], 0.0, -parameters.dynamic['gravity']);
      gl.uniform1f(updateParticleUniforms['u_viscosityCoef'], parameters.dynamic['viscosity coef']);
      gl.uniform1f(updateParticleUniforms['u_restitutionCoef'], parameters.dynamic['restitution coef']);
      gl.uniform1f(updateParticleUniforms['u_frictionCoef'], parameters.dynamic['friction coef']);
      gl.uniform1f(updateParticleUniforms['u_deltaTime'], deltaTime);
      gl.bindVertexArray(fillViewportVao);
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      swapParticleFbObj();
    };

    const renderParticles = () => {
      gl.viewport(0.0, 0.0, canvas.width, canvas.height);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(renderParticleProgram);
      setUniformTexture(gl, 0, particleFbObjR.positionTexture, renderParticleUniforms['u_positionTexture']);
      setUniformTexture(gl, 1, particleFbObjR.velocityTexture, renderParticleUniforms['u_velocityTexture']);
      gl.uniform2f(renderParticleUniforms['u_viewportSize'], canvas.width, canvas.height);
      gl.uniform2f(renderParticleUniforms['u_simulationSpace'], simulationSpace.x, simulationSpace.y);
      gl.uniform1f(renderParticleUniforms['u_targetSpace'], targetSpace);
      gl.uniform1f(renderParticleUniforms['u_particleSize'], parameters.dynamic['particle size']);
      gl.drawArrays(gl.POINT, 0, particleNum);
    };

    const stepSimulation = (deltaTime) => {
      constructBuckets();
      computeDensityAndPressure();
      updateParticles(deltaTime);
    }

    initializeParticles();
    let simulationSeconds = 0.0;
    let previousRealSeconds = performance.now() * 0.001;
    const render = () => {
      stats.update();

      const timeScale = parameters.dynamic['time scale'];
      const currentRealSeconds = performance.now() * 0.001;
      const nextSimulationSeconds = simulationSeconds + timeScale * Math.min(0.05, currentRealSeconds - previousRealSeconds);
      previousRealSeconds = currentRealSeconds;

      const timeStep = parameters.dynamic['time step'];
      while(nextSimulationSeconds - simulationSeconds > timeStep) {
        stepSimulation(timeStep);
        simulationSeconds += timeStep;
      }

      renderParticles();

      requestId = requestAnimationFrame(render);
    }
    render();
  };
  reset();

}());