package com.vc.deg.impl;

import static java.nio.file.StandardCopyOption.REPLACE_EXISTING;
import static java.nio.file.StandardOpenOption.CREATE;
import static java.nio.file.StandardOpenOption.TRUNCATE_EXISTING;
import static java.nio.file.StandardOpenOption.WRITE;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.KryoException;
import com.esotericsoftware.kryo.unsafe.UnsafeInput;
import com.esotericsoftware.kryo.unsafe.UnsafeOutput;
import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.GraphDesigner;
import com.vc.deg.GraphNavigator;
import com.vc.deg.SearchResult;
import com.vc.deg.impl.designer.EvenRegularGraphDesigner;
import com.vc.deg.impl.graph.WeightedUndirectedGraph;
import com.vc.deg.impl.navigation.EvenRegularGraphNavigator;

/**
 * This class wraps the main functions of the library
 *  
 * @author Nico Hezel
 */
public class DynamicExplorationGraph implements com.vc.deg.DynamicExplorationGraph {

	protected WeightedUndirectedGraph internalGraph;
	protected EvenRegularGraphNavigator navigator;
	protected EvenRegularGraphDesigner designer;
	
	
	/**
	 * Use a custom repository. This will be filled when the {@link #add(int, Object)} method is called.
	 * Customs repos can have different storing and data access functions
	 * 
	 * @param objectRepository
	 */
	public DynamicExplorationGraph(FeatureSpace space, int edgesPerNode) {
		this.internalGraph = new WeightedUndirectedGraph(edgesPerNode, space);
		this.designer = new EvenRegularGraphDesigner(internalGraph); 
		this.navigator = new EvenRegularGraphNavigator(internalGraph); 
	}
	
	@Override
	public SearchResult search(FeatureVector query, int k, float eps) {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public GraphNavigator navigator() {
		return navigator;
	}
	
	@Override
	public GraphDesigner designer() {
		return designer;
	}
	
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
}
