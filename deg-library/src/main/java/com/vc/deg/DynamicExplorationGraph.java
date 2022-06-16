package com.vc.deg;

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
import com.vc.deg.data.DataRepository;
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
	protected DataRepository<T> objectRepository;
	protected EvenRegularGraphNavigator navigator;
	protected EvenRegularGraphDesigner designer;
	
	/**
	 * Define how the nodes of the graph will be compared.
	 * 
	 * @param space
	 */
	public DynamicExplorationGraph(FeatureSpace<T> space, int edgesPerNode) {
		this(new FeatureRepository<>(space), edgesPerNode); 
	}
	
	/**
	 * Use a custom repository. This will be filled when the {@link #add(int, Object)} method is called.
	 * Customs repos can have different storing and data access functions
	 * 
	 * @param objectRepository
	 */
	public DynamicExplorationGraph(DataRepository<T> objectRepository, int edgesPerNode) {
		this.objectRepository = objectRepository;
		this.interalGraph = new EvenRegularWeightedUndirectedGraph(edgesPerNode);
		this.designer = new EvenRegularGraphDesigner(interalGraph, objectRepository.getComparator()); 
		this.navigator = new EvenRegularGraphNavigator(interalGraph, objectRepository.getComparator()); 
	}
	
	public void add(int id, T data) {
		this.objectRepository.add(id, data);
		this.designer.add(id);
	}
	
	public void writeToFile(Path file) throws KryoException, IOException {
		Path tempFile = Paths.get(file.getParent().toString(), "~$" + file.getFileName().toString());
		Kryo kryo = new Kryo();
		try(UnsafeOutput output = new UnsafeOutput(Files.newOutputStream(tempFile, TRUNCATE_EXISTING, CREATE, WRITE))) {
			this.objectRepository.write(kryo, output);
			this.interalGraph.write(kryo, output);
		}
		Files.move(tempFile, file, REPLACE_EXISTING);
	}
	
	public void readFromFile(Path file) throws KryoException, IOException {
		Kryo kryo = new Kryo();
		try(UnsafeInput input = new UnsafeInput(Files.newInputStream(file))) {
			this.objectRepository.read(kryo, input);
			this.interalGraph.read(kryo, input);
		} 
	}
}
