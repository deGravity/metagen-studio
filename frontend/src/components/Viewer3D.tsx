import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid, Environment } from '@react-three/drei';
import { useMemo } from 'react';
import * as THREE from 'three';
import type { MeshData } from '../types';

interface Props {
  mesh: MeshData | null;
}

function Mesh({ mesh }: { mesh: MeshData }) {
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(mesh.vertices, 3));
    g.setIndex(new THREE.BufferAttribute(mesh.triangles, 1));
    g.computeVertexNormals();
    g.center();
    g.computeBoundingSphere();
    return g;
  }, [mesh]);

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial color="#6aa3ff" metalness={0.1} roughness={0.6} side={THREE.DoubleSide} />
    </mesh>
  );
}

export function Viewer3D({ mesh }: Props) {
  return (
    <div style={{ width: '100%', height: '100%', background: '#1e1e1e' }}>
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
        {mesh && <Mesh mesh={mesh} />}
        <OrbitControls makeDefault enableDamping dampingFactor={0.1} />
        <axesHelper args={[1.5]} />
      </Canvas>
    </div>
  );
}
