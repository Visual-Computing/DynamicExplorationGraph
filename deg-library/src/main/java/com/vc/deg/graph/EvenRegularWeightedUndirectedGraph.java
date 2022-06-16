package com.vc.deg.graph;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.KryoSerializable;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import com.koloboke.collect.map.IntFloatMap;
import com.koloboke.collect.map.IntObjMap;
import com.koloboke.collect.map.hash.HashIntFloatMaps;
import com.koloboke.collect.map.hash.HashIntObjMaps;
import com.koloboke.collect.set.IntSet;
import com.vc.deg.kryo.HashIntIntFloatMapSerializer;

/**
 * Der Interne Graph mit seinen addNode,deleteNode,addEdge,deleteEdge,serialize Funktionen.
 * Stellt keine Such oder Navigationsmöglichen.
 * 
 * @author Neiko
 *
 */
public class EvenRegularWeightedUndirectedGraph implements KryoSerializable {

	protected final IntObjMap<IntFloatMap> nodes;
	protected int edgesPerNode;
	
	public EvenRegularWeightedUndirectedGraph(int edgesPerNode) {
		this.edgesPerNode = edgesPerNode;
		this.nodes = HashIntObjMaps.newMutableMap();
		
	}

	public EvenRegularWeightedUndirectedGraph(int edgesPerNode, int expectedSize) {
		this.edgesPerNode = edgesPerNode;
		this.nodes = HashIntObjMaps.newMutableMap(expectedSize);		
	}
	
	public void addNode(int id) {
		nodes.put(id, HashIntFloatMaps.newMutableMap(edgesPerNode));
	}
	
	public IntSet getEdgeIds(int id) {
		return nodes.get(id).keySet();
	}

	@Override
	public void write(Kryo kryo, Output output) {
		HashIntIntFloatMapSerializer.store(kryo, output, nodes);
		
	}

	@Override
	public void read(Kryo kryo, Input input) {
		HashIntIntFloatMapSerializer.loadMutable(kryo, input, nodes);
	}
}
