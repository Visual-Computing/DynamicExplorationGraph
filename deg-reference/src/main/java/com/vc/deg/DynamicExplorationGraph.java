package com.vc.deg;

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

import com.vc.deg.data.FeatureRepository;
import com.vc.deg.data.FeatureSpace;
import com.vc.deg.designer.EvenRegularGraphDesigner;
import com.vc.deg.graph.EvenRegularWeightedUndirectedGraph;
import com.vc.deg.navigation.EvenRegularGraphNavigator;

/**
 * This class wraps the main functions of the library
 * 
 * @author Nico Hezel
 *
 * @param <T>
 */
public class DynamicExplorationGraph<T> {

	protected EvenRegularWeightedUndirectedGraph interalGraph;
	protected FeatureRepository<T> objectRepository;
	protected EvenRegularGraphNavigator<T> navigator;
	protected EvenRegularGraphDesigner<T> designer;
	
	/**
	 * Define how the nodes of the graph will be compared and how many edges each node has.
	 * 
	 * @param space
	 * @param edgesPerNode
	 */
	public DynamicExplorationGraph(FeatureSpace<T> space, int edgesPerNode) {
		this.objectRepository = new FeatureRepository<>(space);
		this.interalGraph = new EvenRegularWeightedUndirectedGraph(edgesPerNode);
		this.designer = new EvenRegularGraphDesigner<>(interalGraph, objectRepository); 
		this.navigator = new EvenRegularGraphNavigator<>(interalGraph, objectRepository); 
	}
	
	public void add(int id, T data) {
		this.objectRepository.add(id, data);
		this.designer.add(id);
	}
	
	public void writeToFile(Path file) throws ClassNotFoundException, IOException {
		Path tempFile = Paths.get(file.getParent().toString(), "~$" + file.getFileName().toString());
		
		try(OutputStream os = Files.newOutputStream(tempFile, TRUNCATE_EXISTING, CREATE, WRITE)) {
			ObjectOutputStream output  = new ObjectOutputStream(os);
			this.objectRepository.writeObject(output);
			this.interalGraph.writeObject(output);
		}
		Files.move(tempFile, file, REPLACE_EXISTING);
	}
	
	public void readFromFile(Path file) throws ClassNotFoundException, IOException {
		try(InputStream os = Files.newInputStream(file)) {
			ObjectInputStream input  = new ObjectInputStream(os);
			this.objectRepository.readObject(input);
			this.interalGraph.readObject(input);
		} 
	}
}
