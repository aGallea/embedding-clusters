import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Billboard, Html } from '@react-three/drei'
import * as THREE from 'three'
import { usePlotStore } from '../../stores/plotStore'

const MARKER_COLOR = '#ff3366'
const PULSE_SPEED = 2.5
const PULSE_SCALE_RANGE = 0.08
const BASE_SPHERE_RADIUS = 0.12
const GLOW_SIZE = 0.6

function createGlowTexture(): THREE.CanvasTexture {
  const size = 128
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')!
  const gradient = ctx.createRadialGradient(
    size / 2, size / 2, 0,
    size / 2, size / 2, size / 2,
  )
  gradient.addColorStop(0, 'rgba(255, 51, 102, 0.6)')
  gradient.addColorStop(0.4, 'rgba(255, 51, 102, 0.2)')
  gradient.addColorStop(1, 'rgba(255, 51, 102, 0)')
  ctx.fillStyle = gradient
  ctx.fillRect(0, 0, size, size)
  const texture = new THREE.CanvasTexture(canvas)
  texture.needsUpdate = true
  return texture
}

export default function QueryMarker() {
  const queryPoint = usePlotStore((state) => state.queryPoint)
  const groupRef = useRef<THREE.Group>(null!)
  const materialRef = useRef<THREE.MeshStandardMaterial>(null!)

  const glowTexture = useMemo(() => createGlowTexture(), [])

  useFrame(({ clock }) => {
    if (!groupRef.current || !queryPoint) return
    const t = clock.getElapsedTime()
    const pulse = 1 + Math.sin(t * PULSE_SPEED) * PULSE_SCALE_RANGE
    groupRef.current.scale.setScalar(pulse)

    if (materialRef.current) {
      materialRef.current.emissiveIntensity =
        0.6 + Math.sin(t * PULSE_SPEED) * 0.3
    }
  })

  if (!queryPoint) return null

  return (
    <group
      ref={groupRef}
      position={[queryPoint.x, queryPoint.y, queryPoint.z]}
    >
      {/* Core sphere */}
      <mesh>
        <sphereGeometry args={[BASE_SPHERE_RADIUS, 16, 16]} />
        <meshStandardMaterial
          ref={materialRef}
          color={MARKER_COLOR}
          emissive={MARKER_COLOR}
          emissiveIntensity={0.6}
          toneMapped={false}
        />
      </mesh>

      {/* Glow halo */}
      <Billboard>
        <mesh>
          <planeGeometry args={[GLOW_SIZE, GLOW_SIZE]} />
          <meshBasicMaterial
            map={glowTexture}
            transparent
            blending={THREE.AdditiveBlending}
            depthWrite={false}
            toneMapped={false}
          />
        </mesh>
      </Billboard>

      {/* Label */}
      <Html
        center
        distanceFactor={15}
        style={{ pointerEvents: 'none' }}
        position={[0, BASE_SPHERE_RADIUS + 0.15, 0]}
      >
        <div className="bg-gray-900/80 text-white text-xs px-2 py-0.5 rounded whitespace-nowrap backdrop-blur-sm">
          Query
        </div>
      </Html>
    </group>
  )
}
