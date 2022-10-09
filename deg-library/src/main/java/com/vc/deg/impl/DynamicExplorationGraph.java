package com.vc.deg.impl;

import static java.nio.file.StandardCopyOption.REPLACE_EXISTING;
import static java.nio.file.StandardOpenOption.CREATE;
import static java.nio.file.StandardOpenOption.TRUNCATE_EXISTING;
import static java.nio.file.StandardOpenOption.WRITE;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collection;
import java.util.Random;
import java.util.function.IntConsumer;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.KryoException;
import com.esotericsoftware.kryo.unsafe.UnsafeInput;
import com.esotericsoftware.kryo.unsafe.UnsafeOutput;
import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.graph.GraphDesigner;
import com.vc.deg.graph.GraphFilter;
import com.vc.deg.graph.VertexConsumer;
import com.vc.deg.impl.designer.EvenRegularGraphDesigner;
import com.vc.deg.impl.graph.WeightedUndirectedRegularGraph;

/**
 * This class wraps the main functions of the library
 *  
 * @author Nico Hezel
 */
public class DynamicExplorationGraph implements com.vc.deg.DynamicExplorationGraph {

	protected WeightedUndirectedRegularGraph internalGraph;
	protected EvenRegularGraphDesigner designer;
	
	
	/**
	 * Use a custom repository. This will be filled when the {@link #add(int, Object)} method is called.
	 * Customs repos can have different storing and data access functions
	 * 
	 * @param objectRepository
	 */
	public DynamicExplorationGraph(FeatureSpace space, int edgesPerNode) {
		this.internalGraph = new WeightedUndirectedRegularGraph(edgesPerNode, space);
		this.designer = new EvenRegularGraphDesigner(internalGraph); 
	}
	
	
	@Override
	public GraphDesigner designer() {
		return designer;
	}
	
	@Override
	public void writeToFile(Path file) throws KryoException, IOException {
		Path tempFile = Paths.get(file.getParent().toString(), "~$" + file.getFileName().toString());
		Kryo kryo = new Kryo();
		try(UnsafeOutput output = new UnsafeOutput(Files.newOutputStream(tempFile, TRUNCATE_EXISTING, CREATE, WRITE))) {
			this.internalGraph.write(kryo, output);
		}
		Files.move(tempFile, file, REPLACE_EXISTING);
	}
	
	public void readFromFile(Path file) throws KryoException, IOException {
		Kryo kryo = new Kryo();
		try(UnsafeInput input = new UnsafeInput(Files.newInputStream(file))) {
			this.internalGraph.read(kryo, input);
		} 
	}

	@Override
	public DynamicExplorationGraph copy() {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public int[] search(Collection<FeatureVector> queries, int k, float eps, GraphFilter filter) {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public int[] explore(int[] entryLabel, int k, int maxDistanceComputationCount, GraphFilter filter) {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public boolean hasLabel(int label) {
		// TODO Auto-generated method stub
		return false;
	}


	@Override
	public FeatureSpace getFeatureSpace() {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public FeatureVector getFeature(int label) {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public int size() {
		// TODO Auto-generated method stub
		return 0;
	}


	@Override
	public void forEachVertex(VertexConsumer consumer) {
		// TODO Auto-generated method stub
		
	}


	@Override
	public void forEachNeighbor(int label, IntConsumer idConsumer) {
		// TODO Auto-generated method stub
		
	}


	@Override
	public int getRandomLabel(Random random) {
		// TODO Auto-generated method stub
		return 0;
	}
}
