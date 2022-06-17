package com.vc.deg.ref;

import static java.nio.file.StandardCopyOption.REPLACE_EXISTING;
import static java.nio.file.StandardOpenOption.CREATE;
import static java.nio.file.StandardOpenOption.TRUNCATE_EXISTING;
import static java.nio.file.StandardOpenOption.WRITE;

import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import com.vc.deg.FeatureSpace;
import com.vc.deg.ref.designer.EvenRegularGraphDesigner;
import com.vc.deg.ref.graph.EvenRegularWeightedUndirectedGraph;
import com.vc.deg.ref.navigation.EvenRegularGraphNavigator;

/**
 * This class wraps the main functions of the library
 * 
 * @author Nico Hezel
 */
public class DynamicExplorationGraph implements com.vc.deg.DynamicExplorationGraph {

	protected EvenRegularWeightedUndirectedGraph internalGraph;
	protected EvenRegularGraphNavigator navigator;
	protected EvenRegularGraphDesigner designer;
	
	/**
	 * Define how the nodes of the graph will be compared and how many edges each node has.
	 * 
	 * @param space
	 * @param edgesPerNode
	 */
	public DynamicExplorationGraph(FeatureSpace space, int edgesPerNode) {
		this.internalGraph = new EvenRegularWeightedUndirectedGraph(edgesPerNode, space);
		this.designer = new EvenRegularGraphDesigner(internalGraph); 
		this.navigator = new EvenRegularGraphNavigator(internalGraph); 
	}
	
	@Override
	public EvenRegularGraphNavigator navigator() {
		return navigator;
	}
	
	@Override
	public EvenRegularGraphDesigner designer() {
		return designer;
	}
	
	public void writeToFile(Path file) throws ClassNotFoundException, IOException {
		Path tempFile = Paths.get(file.getParent().toString(), "~$" + file.getFileName().toString());
		
		try(OutputStream os = Files.newOutputStream(tempFile, TRUNCATE_EXISTING, CREATE, WRITE)) {
			ObjectOutputStream output = new ObjectOutputStream(os);
			this.internalGraph.writeObject(output);
		}
		Files.move(tempFile, file, REPLACE_EXISTING);
	}
	
	public void readFromFile(Path file) throws ClassNotFoundException, IOException {
		try(InputStream os = Files.newInputStream(file)) {
			ObjectInputStream input = new ObjectInputStream(os);
			this.internalGraph.readObject(input);
		} 
	}
}
