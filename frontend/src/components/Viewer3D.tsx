import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid, Environment } from '@react-three/drei';
import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import type { MeshData } from '../types';

interface Props {
  mesh: MeshData | null;
}

const MAX_TILE = 6;
const MAX_INSTANCES = MAX_TILE ** 3;

function TiledMesh({ mesh, tileCount }: { mesh: MeshData; tileCount: number }) {
  const ref = useRef<THREE.InstancedMesh>(null);

  const { geometry, extent } = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(mesh.vertices, 3));
    g.setIndex(new THREE.BufferAttribute(mesh.triangles, 1));
    g.computeVertexNormals();
    g.center();
    g.computeBoundingBox();
    const ext = new THREE.Vector3();
    g.boundingBox!.getSize(ext);
    return { geometry: g, extent: ext };
  }, [mesh]);

  useEffect(() => {
    const im = ref.current;
    if (!im) return;
    const m = new THREE.Matrix4();
    const N = tileCount;
    const offset = (N - 1) / 2;
    let i = 0;
    for (let x = 0; x < N; x++) {
      for (let y = 0; y < N; y++) {
        for (let z = 0; z < N; z++) {
          m.makeTranslation(
            (x - offset) * extent.x,
            (y - offset) * extent.y,
            (z - offset) * extent.z,
          );
          im.setMatrixAt(i++, m);
        }
      }
    }
    im.count = i;
    im.instanceMatrix.needsUpdate = true;
    im.computeBoundingSphere();
  }, [tileCount, extent]);

  return (
    <instancedMesh
      ref={ref}
      args={[undefined, undefined, MAX_INSTANCES]}
      geometry={geometry}
    >
      <meshStandardMaterial
        color="#6aa3ff"
        metalness={0.1}
        roughness={0.6}
        side={THREE.DoubleSide}
      />
    </instancedMesh>
  );
}

export function Viewer3D({ mesh }: Props) {
  const [tileCount, setTileCount] = useState(1);

  return (
    <div style={{ width: '100%', height: '100%', background: '#1e1e1e', position: 'relative' }}>
      <Canvas camera={{ position: [2, 2, 2], fov: 50 }} dpr={[1, 2]}>
        <ambientLight intensity={0.4} />
        <directionalLight position={[5, 5, 5]} intensity={0.8} />
        <directionalLight position={[-5, -3, -5]} intensity={0.3} />
        <Environment preset="studio" />
        <Grid
          args={[10, 10]}
          cellSize={0.1}
          cellColor="#404040"
          sectionSize={1}
          sectionColor="#606060"
          fadeDistance={20}
          infiniteGrid
        />
        {mesh && <TiledMesh mesh={mesh} tileCount={tileCount} />}
        <OrbitControls makeDefault enableDamping dampingFactor={0.1} />
        <axesHelper args={[1.5]} />
      </Canvas>
      <div
        style={{
          position: 'absolute',
          top: 8,
          left: 8,
          background: 'rgba(30,30,30,0.85)',
          padding: '8px 12px',
          borderRadius: 6,
          color: '#ddd',
          fontSize: 12,
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          userSelect: 'none',
        }}
      >
        <label htmlFor="tile-slider">tile</label>
        <input
          id="tile-slider"
          type="range"
          min={1}
          max={MAX_TILE}
          step={1}
          value={tileCount}
          onChange={(e) => setTileCount(parseInt(e.target.value, 10))}
          style={{ width: 90 }}
        />
        <span style={{ minWidth: 44, textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
          {tileCount}×{tileCount}×{tileCount}
        </span>
      </div>
    </div>
  );
}
