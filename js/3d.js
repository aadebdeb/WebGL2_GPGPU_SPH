(function() {

const INITIALIZE_BUCKET_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

out ivec2 o_bucket;

uniform sampler2D u_positionTexture;
uniform float u_bucketSize;
uniform int u_particleNum;
uniform ivec3 u_bucketNum;

int convertCoordToIndex(ivec2 coord, int sizeX) {
  return coord.x + sizeX * coord.y;
}

int getBucketIndex(vec3 position) {
  ivec3 bucketCoord = ivec3(position / u_bucketSize);
  return bucketCoord.x + bucketCoord.y * u_bucketNum.x + bucketCoord.z * (u_bucketNum.x * u_bucketNum.y);
}

void main(void) {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  int particleTextureSizeX = textureSize(u_positionTexture, 0).x;
  int particleIndex = convertCoordToIndex(coord, particleTextureSizeX);
  if (particleIndex >= u_particleNum) {
    o_bucket = ivec2(65535, 65535);
    return;
  }
  vec3 position = texelFetch(u_positionTexture, coord, 0).xyz;
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
uniform ivec3 u_bucketNum;

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
  int maxBucketIndex = u_bucketNum.x * u_bucketNum.y * u_bucketNum.z;

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

layout (location = 0) out vec3 o_position;
layout (location = 1) out vec3 o_velocity;

uniform int u_particleNum;
uniform int u_particleTextureSizeX;
uniform float u_targetSpace;
uniform ivec2 u_damParticleNumXY;

int convertCoordToIndex(ivec2 coord, int sizeX) {
  return coord.x + sizeX * coord.y;
}

void main(void) {
  int index = convertCoordToIndex(ivec2(gl_FragCoord.xy), u_particleTextureSizeX);

  if (index >= u_particleNum) { // unused pixels
    o_position = vec3(0.0);
    o_velocity = vec3(0.0);
    return;
  }

  ivec3 posIndex= ivec3(index % u_damParticleNumXY.x, index % (u_damParticleNumXY.x * u_damParticleNumXY.y) / u_damParticleNumXY.x, index / (u_damParticleNumXY.x * u_damParticleNumXY.y));

  float halfTargetSpace = 0.5 * u_targetSpace;

  vec3 posOffset = posIndex.y % 2 == 0 ?
    vec3(halfTargetSpace, halfTargetSpace, halfTargetSpace) :
    vec3(u_targetSpace, halfTargetSpace, u_targetSpace);

  o_position = vec3(posIndex) * u_targetSpace + posOffset;
  o_velocity = vec3(0.0);
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
uniform ivec3 u_bucketNum;

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
  StdKernelCoef = 315.0 / (64.0 * PI * u_kernelRadius * u_kernelRadius * u_kernelRadius * u_kernelRadius * u_kernelRadius * u_kernelRadius * u_kernelRadius * u_kernelRadius * u_kernelRadius);
}

float calcStdKernel(float d) {
  if (d < u_kernelRadius) {
    float x = u_kernelRadius * u_kernelRadius - d * d;
    return StdKernelCoef * x * x * x;
  }
  return 0.0;
}

float computeDensityFromBucket(int particleIndex, vec3 position, ivec3 bucketPosition, ivec3 bucketNum, int particleTextureSizeX, int bucketReferrerTextureSizeX) {
  if (bucketPosition.x < 0 || bucketPosition.x >= bucketNum.x ||
      bucketPosition.y < 0 || bucketPosition.y >= bucketNum.y ||
      bucketPosition.z < 0 || bucketPosition.z >= bucketNum.z) {
    return 0.0;
  }
  int bucketIndex = bucketPosition.x + bucketPosition.y * bucketNum.x + bucketPosition.z * (bucketNum.x * bucketNum.y);
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
    vec3 otherPos = texelFetch(u_positionTexture, convertIndexToCoord(otherIndex, particleTextureSizeX), 0).xyz;
    float dist = length(position - otherPos);
    density += calcStdKernel(dist);
  }
  return density;
}

float computeDensity(int particleIndex, vec3 position, int particleTextureSizeX) {
  int bucketReferrerTextureSizeX = textureSize(u_bucketReferrerTexture, 0).x;

  vec3 bucketPosition = position / (2.0 * u_kernelRadius);
  int xOffset = fract(bucketPosition.x) < 0.5 ? -1 : 1;
  int yOffset = fract(bucketPosition.y) < 0.5 ? -1 : 1;
  int zOffset = fract(bucketPosition.z) < 0.5 ? -1 : 1;

  ivec3 bucketPosition000 = ivec3(bucketPosition);
  ivec3 bucketPosition100 = bucketPosition000 + ivec3(xOffset, 0, 0);
  ivec3 bucketPosition010 = bucketPosition000 + ivec3(0, yOffset, 0);
  ivec3 bucketPosition110 = bucketPosition000 + ivec3(xOffset, yOffset, 0);
  ivec3 bucketPosition001 = bucketPosition000 + ivec3(0, 0, zOffset);
  ivec3 bucketPosition101 = bucketPosition000 + ivec3(xOffset, 0, zOffset);
  ivec3 bucketPosition011 = bucketPosition000 + ivec3(0, yOffset, zOffset);
  ivec3 bucketPosition111 = bucketPosition000 + ivec3(xOffset, yOffset, zOffset);

  float density = 0.0;
  density += computeDensityFromBucket(particleIndex, position, bucketPosition000, u_bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX);
  density += computeDensityFromBucket(particleIndex, position, bucketPosition100, u_bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX);
  density += computeDensityFromBucket(particleIndex, position, bucketPosition010, u_bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX);
  density += computeDensityFromBucket(particleIndex, position, bucketPosition110, u_bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX);
  density += computeDensityFromBucket(particleIndex, position, bucketPosition001, u_bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX);
  density += computeDensityFromBucket(particleIndex, position, bucketPosition101, u_bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX);
  density += computeDensityFromBucket(particleIndex, position, bucketPosition011, u_bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX);
  density += computeDensityFromBucket(particleIndex, position, bucketPosition111, u_bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX);

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

  vec3 position = texelFetch(u_positionTexture, coord, 0).xyz;
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

layout (location = 0) out vec3 o_position;
layout (location = 1) out vec3 o_velocity;

uniform isampler2D u_bucketTexture;
uniform isampler2D u_bucketReferrerTexture;
uniform int u_particleNum;
uniform ivec3 u_bucketNum;

uniform sampler2D u_positionTexture;
uniform sampler2D u_velocityTexture;
uniform sampler2D u_densityTexture;
uniform sampler2D u_pressureTexture;
uniform vec3 u_simulationSpace;
uniform float u_targetSpace;
uniform float u_kernelRadius;
uniform float u_mass;
uniform vec3 u_gravity;
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
  SpikyKernelDerivativeCoef = -45.0 / (PI * u_kernelRadius * u_kernelRadius * u_kernelRadius * u_kernelRadius * u_kernelRadius * u_kernelRadius);
  ViscosityKernelSecondDerivativeCoef = 45.0 / (PI * u_kernelRadius * u_kernelRadius * u_kernelRadius * u_kernelRadius * u_kernelRadius * u_kernelRadius);
}

float calcSpikyKernelDerivative(float d) {
  if (d < u_kernelRadius) {
    float x = u_kernelRadius - d;
    return SpikyKernelDerivativeCoef * x * x;
  }
  return 0.0;
}

vec3 calcSpikyKernelGradient(vec3 center, vec3 position) {
  vec3 v = center - position;
  if (length(v) == 0.0) {
    return vec3(0.0);
  }
  return calcSpikyKernelDerivative(length(v)) * normalize(v);
}

float calcViscosityKernelSecondDerivative(float d) {
  if (d < u_kernelRadius) {
    return ViscosityKernelSecondDerivativeCoef * (u_kernelRadius - d);
  }
  return 0.0;
}

void accumulatePressureAndViscosityForces(int particleIndex, vec3 position, vec3 velocity, float density, float pressure, ivec3 bucketPosition, ivec3 bucketNum, int particleTextureSizeX, int bucketReferrerTextureSizeX, inout vec3 pressureForce, inout vec3 viscosityForce) {
  if (bucketPosition.x < 0 || bucketPosition.x >= bucketNum.x ||
      bucketPosition.y < 0 || bucketPosition.y >= bucketNum.y ||
      bucketPosition.z < 0 || bucketPosition.z >= bucketNum.z) {
    return;
  }
  int bucketIndex = bucketPosition.x + bucketPosition.y * bucketNum.x + bucketPosition.z * (bucketNum.x * bucketNum.y);
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
    vec3 otherPos = texelFetch(u_positionTexture, otherCoord, 0).xyz;

    float dist = length(position - otherPos);

    if (density != 0.0 && dist < u_kernelRadius) {
      float otherDens = texelFetch(u_densityTexture, otherCoord, 0).x;
      if (otherDens == 0.0) { continue; }
      vec3 otherVel = texelFetch(u_velocityTexture, otherCoord, 0).xyz;
      float otherPres = texelFetch(u_pressureTexture, otherCoord, 0).x;

      float otherDens2 = otherDens * otherDens;
      pressureForce += -(pressure / density2 + otherPres / otherDens2) * calcSpikyKernelGradient(position, otherPos);

      viscosityForce += (otherVel - velocity) * calcViscosityKernelSecondDerivative(dist) / otherDens;
    }
  }
}

vec3 computePressureAndViscosityForces(int particleIndex, vec3 position, vec3 velocity, float density, float pressure, int particleTextureSizeX) {
  int bucketReferrerTextureSizeX = textureSize(u_bucketReferrerTexture, 0).x;

  vec3 bucketPosition = position / (2.0 * u_kernelRadius);
  int xOffset = fract(bucketPosition.x) < 0.5 ? -1 : 1;
  int yOffset = fract(bucketPosition.y) < 0.5 ? -1 : 1;
  int zOffset = fract(bucketPosition.z) < 0.5 ? -1 : 1;

  ivec3 bucketPosition000 = ivec3(bucketPosition);
  ivec3 bucketPosition100 = bucketPosition000 + ivec3(xOffset, 0, 0);
  ivec3 bucketPosition010 = bucketPosition000 + ivec3(0, yOffset, 0);
  ivec3 bucketPosition110 = bucketPosition000 + ivec3(xOffset, yOffset, 0);
  ivec3 bucketPosition001 = bucketPosition000 + ivec3(0, 0, zOffset);
  ivec3 bucketPosition101 = bucketPosition000 + ivec3(xOffset, 0, zOffset);
  ivec3 bucketPosition011 = bucketPosition000 + ivec3(0, yOffset, zOffset);
  ivec3 bucketPosition111 = bucketPosition000 + ivec3(xOffset, yOffset, zOffset);

  vec3 pressureForce = vec3(0.0);
  vec3 viscosityForce = vec3(0.0);
  accumulatePressureAndViscosityForces(particleIndex, position, velocity, density, pressure, bucketPosition000, u_bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX, pressureForce, viscosityForce);
  accumulatePressureAndViscosityForces(particleIndex, position, velocity, density, pressure, bucketPosition100, u_bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX, pressureForce, viscosityForce);
  accumulatePressureAndViscosityForces(particleIndex, position, velocity, density, pressure, bucketPosition010, u_bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX, pressureForce, viscosityForce);
  accumulatePressureAndViscosityForces(particleIndex, position, velocity, density, pressure, bucketPosition110, u_bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX, pressureForce, viscosityForce);
  accumulatePressureAndViscosityForces(particleIndex, position, velocity, density, pressure, bucketPosition001, u_bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX, pressureForce, viscosityForce);
  accumulatePressureAndViscosityForces(particleIndex, position, velocity, density, pressure, bucketPosition101, u_bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX, pressureForce, viscosityForce);
  accumulatePressureAndViscosityForces(particleIndex, position, velocity, density, pressure, bucketPosition011, u_bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX, pressureForce, viscosityForce);
  accumulatePressureAndViscosityForces(particleIndex, position, velocity, density, pressure, bucketPosition111, u_bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX, pressureForce, viscosityForce);
  pressureForce *= u_mass;
  viscosityForce *= u_mass * u_mass * u_viscosityCoef;

  return pressureForce + viscosityForce;
}

vec3 computeExternalForce() {
  return u_mass * u_gravity;
}

vec3 computeForce(int particleIndex, vec3 position, vec3 velocity, float density, float pressure, int particleTextureSizeX) {
  vec3 force = vec3(0.0);
  force += density == 0.0 ? vec3(0.0) : computePressureAndViscosityForces(particleIndex, position, velocity, density, pressure, particleTextureSizeX);
  force += computeExternalForce();
  return force;
}

void collideWall(vec3 center, vec3 normal, inout vec3 position, inout vec3 velocity) {
  vec3 toP = position - center;
  float dotToP = dot(normal, toP);
  vec3 toPN = normal * dotToP;
  vec3 toPT = toP - toPN;
  float dist = length(toPN);
  if (dotToP >= 0.0 && dist > 0.5 * u_targetSpace) {
    return;
  }
  vec3 nearestPos = center + toPT;
  position = nearestPos + 0.5 * u_targetSpace * normal;
  if (dot(velocity, normal) < 0.0) {
    vec3 velN = u_restitutionCoef * (normal * dot(normal, velocity));
    vec3 velT = u_frictionCoef * (velocity - velN);
    velocity = -velN + velT;
  }
}

void solveCollision(inout vec3 position, inout vec3 velocity) {
  vec3 halfSimSpace = 0.5 * u_simulationSpace;
  collideWall(vec3(halfSimSpace.x, 0.0, halfSimSpace.z), vec3(0.0, 1.0, 0.0), position, velocity);
  collideWall(vec3(halfSimSpace.x, halfSimSpace.y, 0.0), vec3(0.0, 0.0, 1.0), position, velocity);
  collideWall(vec3(0.0, halfSimSpace.y, halfSimSpace.z), vec3(1.0, 0.0, 0.0), position, velocity);
  collideWall(vec3(halfSimSpace.x, u_simulationSpace.y, halfSimSpace.z), vec3(0.0, -1.0, 0.0), position, velocity);
  collideWall(vec3(halfSimSpace.x, halfSimSpace.y, u_simulationSpace.z), vec3(0.0, 0.0, -1.0), position, velocity);
  collideWall(vec3(u_simulationSpace.x, halfSimSpace.y, halfSimSpace.z), vec3(-1.0, 0.0, 0.0), position, velocity);

}

void main(void) {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  int particleTextureSizeX = textureSize(u_positionTexture, 0).x;
  int particleIndex = convertCoordToIndex(coord, particleTextureSizeX);
  if (particleIndex >= u_particleNum) {
    o_position = vec3(0.0);
    o_velocity = vec3(0.0);
    return;
  }

  initializeConstants();

  vec3 position = texelFetch(u_positionTexture, coord, 0).xyz;
  vec3 velocity = texelFetch(u_velocityTexture, coord, 0).xyz;
  float density = texelFetch(u_densityTexture, coord, 0).x;
  float pressure = texelFetch(u_pressureTexture, coord, 0).x;

  vec3 force = computeForce(particleIndex, position, velocity, density, pressure, particleTextureSizeX);
  vec3 nextVelocity = velocity + u_deltaTime * force;
  vec3 nextPosition = position + u_deltaTime * nextVelocity;
  solveCollision(nextPosition, nextVelocity);

  o_position = nextPosition;
  o_velocity = nextVelocity;
}
`;

  const RENDER_PARTICLE_VERTEX_SHADER_SOURCE =
`#version 300 es

layout (location = 0) in vec3 i_position;
layout (location = 1) in vec3 i_normal;

out vec3 v_normal;
out vec3 v_color;

uniform sampler2D u_positionTexture;
uniform sampler2D u_velocityTexture;
uniform vec3 u_simulationSpace;
uniform float u_targetSpace;
uniform mat4 u_mvpMatrix;

ivec2 convertIndexToCoord(int index, int sizeX) {
  return ivec2(index % sizeX, index / sizeX);
}

void main(void) {
  ivec2 coord = convertIndexToCoord(gl_InstanceID, textureSize(u_positionTexture, 0).x);
  vec3 instancePosition = texelFetch(u_positionTexture, coord, 0).xyz;
  float scale = 300.0;
  vec3 position = i_position * 0.5 * u_targetSpace * 2.0 * scale  + (instancePosition * 2.0 - u_simulationSpace) * scale;

  gl_Position = u_mvpMatrix * vec4(position, 1.0);
  v_normal = (u_mvpMatrix * vec4(i_normal, 0.0)).xyz;

  vec3 velocity = texelFetch(u_velocityTexture, coord, 0).xyz;
  v_color = mix(vec3(0.1, 1.0, 0.7), vec3(1.0, 0.5, 0.2), smoothstep(0.0, 1.0, length(velocity)));
}
`;

  const RENDER_PARTICLE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

in vec3 v_normal;
in vec3 v_color;

out vec4 o_color;

vec3 LightDir = normalize(vec3(0.0, 1.0, 0.0));

void main(void) {
  vec3 normal = normalize(v_normal);
  vec3 color = v_color * smoothstep(-3.0, 1.0, dot(LightDir, normal)); 
  o_color = vec4(color, 1.0);
}
`;

  const createParticleFramebuffer = function(gl, size) {
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    const positionTexture = createTexture(gl, size, size, gl.RGBA32F, gl.RGBA, gl.FLOAT);
    gl.bindTexture(gl.TEXTURE_2D, positionTexture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, positionTexture, 0);
    const velocityTexture = createTexture(gl, size, size, gl.RGBA32F, gl.RGBA, gl.FLOAT);
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
      const h9 = h8 * this._h;
      this._coef = 315.0 / (64.0 * Math.PI * h9);
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
    let useOffset = false;
    let sum = 0.0;
    for (let y = 0.0; y < kernelRadius; y += targetSpace) {
      const offset = useOffset ? 0.5 * targetSpace : 0.0;
      for (let x = offset; x < kernelRadius; x += targetSpace) {
        for (let z = offset; z < kernelRadius; z += targetSpace) {
          if (!useOffset) {
            if (y !== 0.0) {
              if ((x !== 0.0 && z === 0.0) || (x === 0.0 && z !== 0.0)) {
                sum += 4.0 * kernel.value(new Vector3(x, y, z).mag());
              } else if(x !== 0.0 && z !== 0.0) {
                sum += 8.0 * kernel.value(new Vector3(x, y, z).mag());
              }
            } else {
              if ((x !== 0.0 && z === 0.0) || (x === 0.0 && z !== 0.0)) {
                sum += 2.0 * kernel.value(new Vector3(x, y, z).mag());
              } else if(x !== 0.0 && z !== 0.0) {
                sum += 4.0 * kernel.value(new Vector3(x, y, z).mag());
              }
            }
          } else {
            sum += 8.0 * kernel.value(new Vector3(x, y, z).mag());
          }
        }
      }
      useOffset = !useOffset;
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
      'time step': 0.001,
      'time scale': 0.5,
    },
    static: {
      'dam width': 0.5,
      'dam height': 1.6,
      'dam depth': 0.5,
      'target space': 0.03,
      'kernel scale': 2.1,
    },
    camera: {
      'angle': -65.0,
      'distance': 1500.0,
      'height': 100.0
    },
    'reset': _ => reset()
  };

  const gui = new dat.GUI();
  const dynamicFolder = gui.addFolder('dynamic parameters');
  dynamicFolder.add(parameters.dynamic, 'viscosity coef', 0.0, 2.0).step(0.01);
  dynamicFolder.add(parameters.dynamic, 'restitution coef', 0.0, 1.0).step(0.01);
  dynamicFolder.add(parameters.dynamic, 'friction coef', 0.0, 1.0).step(0.01);
  dynamicFolder.add(parameters.dynamic, 'gravity', 0.0, 19.6).step(0.01);
  dynamicFolder.add(parameters.dynamic, 'time step', 0.0001, 0.002).step(0.0001);
  dynamicFolder.add(parameters.dynamic, 'time scale', 0.0, 1.0).step(0.01);
  const staticFolder =gui.addFolder('static parameters');
  staticFolder.add(parameters.static, 'dam width', 0.3, 1.0).step(0.01);
  staticFolder.add(parameters.static, 'dam height', 0.3, 1.6).step(0.01);
  staticFolder.add(parameters.static, 'dam depth', 0.3, 1.0).step(0.01);
  staticFolder.add(parameters.static, 'target space', 0.005, 0.1).step(0.0001);
  staticFolder.add(parameters.static, 'kernel scale', 1.5, 3.5).step(0.01);
  const cameraFolder = gui.addFolder('camera');
  cameraFolder.add(parameters.camera, 'angle', -180, 180);
  cameraFolder.add(parameters.camera, 'distance', 50.0, 3000.0);
  cameraFolder.add(parameters.camera, 'height', 0.0, 1000.0);
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
  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.CULL_FACE);

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
    'u_particleNum', 'u_particleTextureSizeX', 'u_targetSpace', 'u_damParticleNumXY'
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
    'u_positionTexture', 'u_simulationSpace', 'u_targetSpace', 'u_mvpMatrix', 'u_velocityTexture'
  ]);

  const fillViewportVao = createFillViewportVao(gl);

  const sphereMesh = createSphere(1.0, 6, 8);
  const sphereVao = createVao(gl, [
    { buffer: createVbo(gl, sphereMesh.positions), index: 0, size: 3 },
    { buffer: createVbo(gl, sphereMesh.normals), index: 1, size: 3}
  ], createIbo(gl, sphereMesh.indices));

  let requestId = null;
  const reset = () => {
    if (requestId !== null) {
      cancelAnimationFrame(requestId);
    }

    const damWidth = parameters.static['dam width'];
    const damHeight = parameters.static['dam height'];
    const damDepth = parameters.static['dam depth'];
    const targetSpace = parameters.static['target space'];
    const simulationSpace = new Vector3(1.2, 1.6, 1.4);
    const kernelRadius = targetSpace * parameters.static['kernel scale'];
    const bucketSize = 2.0 * kernelRadius;
    const targetDensity = 1000.0;
    const mass = calcMass(kernelRadius, targetSpace, targetDensity);
    const speedOfSound = 1481.0; // (m/s in 20 degree)
    const eosExponent = 7.0;
    const negativePressureScale = 0.0;
    const bucketNum = Vector3.div(simulationSpace, bucketSize).ceil().add(new Vector3(1, 1, 1));
    const totalBuckets = bucketNum.x * bucketNum.y * bucketNum.z;

    const halfTargetSpace = 0.5 * targetSpace;

    const damParticleNum = new Vector3(
      Math.floor((damWidth - halfTargetSpace) / targetSpace),
      Math.floor((damHeight - halfTargetSpace) / targetSpace),
      Math.floor((damDepth - halfTargetSpace) / targetSpace)
    );
    const particleNum = damParticleNum.x * damParticleNum.y * damParticleNum.z;
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
      gl.uniform2i(initializeParticleUniforms['u_damParticleNumXY'], damParticleNum.x, damParticleNum.y);
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
      gl.uniform3i(initializeBucketUniforms['u_bucketNum'], bucketNum.x, bucketNum.y, bucketNum.z);
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
      gl.uniform3i(initializeBucketReferrerUniforms['u_bucketNum'], bucketNum.x, bucketNum.y, bucketNum.z);
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
      gl.uniform3i(computeDensityAndPressureUniforms['u_bucketNum'], bucketNum.x, bucketNum.y, bucketNum.z);
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
      gl.uniform3i(updateParticleUniforms['u_bucketNum'], bucketNum.x, bucketNum.y, bucketNum.z);
      setUniformTexture(gl, 2, particleFbObjR.positionTexture, updateParticleUniforms['u_positionTexture']);
      setUniformTexture(gl, 3, particleFbObjR.velocityTexture, updateParticleUniforms['u_velocityTexture']);
      setUniformTexture(gl, 4, densityAndPressureFbObj.densityTexture, updateParticleUniforms['u_densityTexture']);
      setUniformTexture(gl, 5, densityAndPressureFbObj.pressureTexture, updateParticleUniforms['u_pressureTexture']);
      gl.uniform3f(updateParticleUniforms['u_simulationSpace'], simulationSpace.x, simulationSpace.y, simulationSpace.z);
      gl.uniform1f(updateParticleUniforms['u_targetSpace'], targetSpace);
      gl.uniform1f(updateParticleUniforms['u_kernelRadius'], kernelRadius);
      gl.uniform1f(updateParticleUniforms['u_mass'], mass);
      gl.uniform3f(updateParticleUniforms['u_gravity'], 0.0, -parameters.dynamic['gravity'], 0.0);
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
      const cameraRadian = Math.PI * parameters.camera['angle'] / 180.0;
      const cameraPosition = new Vector3(
        parameters.camera['distance'] * Math.cos(cameraRadian),
        parameters.camera['height'],
        parameters.camera['distance'] * Math.sin(cameraRadian));
      const viewMatrix = Matrix4.inverse(Matrix4.lookAt(
        cameraPosition, Vector3.zero, new Vector3(0.0, 1.0, 0.0)
      ));
      const projectionMatrix = Matrix4.perspective(canvas.width / canvas.height, 60, 0.01, 5000.0);
      const mvpMatrix = Matrix4.mul(viewMatrix, projectionMatrix);

      gl.viewport(0.0, 0.0, canvas.width, canvas.height);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.useProgram(renderParticleProgram);
      setUniformTexture(gl, 0, particleFbObjR.positionTexture, renderParticleUniforms['u_positionTexture']);
      gl.uniform3f(renderParticleUniforms['u_simulationSpace'], simulationSpace.x, simulationSpace.y, simulationSpace.z);
      gl.uniform1f(renderParticleUniforms['u_targetSpace'], targetSpace);
      gl.uniformMatrix4fv(renderParticleUniforms['u_mvpMatrix'], false, mvpMatrix.elements);
      setUniformTexture(gl, 1, particleFbObjR.velocityTexture, renderParticleUniforms['u_velocityTexture']);
      gl.bindVertexArray(sphereVao);
      gl.drawElementsInstanced(gl.TRIANGLES, sphereMesh.indices.length, gl.UNSIGNED_SHORT, 0, particleNum);
      gl.bindVertexArray(null);
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