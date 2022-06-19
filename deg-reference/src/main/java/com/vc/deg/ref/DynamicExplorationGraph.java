package com.vc.deg.ref;

import static java.nio.file.StandardCopyOption.REPLACE_EXISTING;
import static java.nio.file.StandardOpenOption.CREATE;
import static java.nio.file.StandardOpenOption.TRUNCATE_EXISTING;
import static java.nio.file.StandardOpenOption.WRITE;

import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import com.vc.deg.FeatureSpace;
import com.vc.deg.io.LittleEndianDataInputStream;
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

	/**
	 * Define how the nodes of the graph will be compared and how many edges each node has.
	 * 
	 * @param space
	 * @param expectedSize
	 * @param edgesPerNode
	 */
	public DynamicExplorationGraph(FeatureSpace space, int expectedSize, int edgesPerNode) {
		this.internalGraph = new EvenRegularWeightedUndirectedGraph(edgesPerNode, expectedSize, space);
		this.designer = new EvenRegularGraphDesigner(internalGraph); 
		this.navigator = new EvenRegularGraphNavigator(internalGraph); 
	}
	
	/**
	 * Used by the copy constructor and read file methods
	 * 
	 * @param graph
	 */
	private DynamicExplorationGraph(EvenRegularWeightedUndirectedGraph graph) {
		this.internalGraph = graph;
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



	/**
	 * Goal: read and write should be compatible with deglib in c-lang. Problem is the endians and the FeatureVector object type written into the output file.
	 * 		 We might want to store another file next to it with the FeatureVector object type information (e.g. storing a single FeatureVector object).
	 * 		 - we use the FeatureVector and memory pool option to when reading from file 
	 * 		 - we could write primitive types in the filename (good for c-lang too) 
	 * 		 - if type in filename is object, we ask FeatureFactory.getCustomFactory()
	 * 
	 * 
	 * 
	 * @param file
	 * @throws ClassNotFoundException
	 * @throws IOException
	 */
	public void writeToFile(Path file) throws ClassNotFoundException, IOException {
		Path tempFile = Paths.get(file.getParent().toString(), "~$" + file.getFileName().toString());

		try(OutputStream os = Files.newOutputStream(tempFile, TRUNCATE_EXISTING, CREATE, WRITE)) {
			// TODO support littleEndian files to be compatible with deglib in c-lang
			// https://github.com/google/guava/blob/master/guava/src/com/google/common/io/LittleEndianDataOutputStream.java
			// https://stackoverflow.com/questions/7024039/in-java-when-writing-to-a-file-with-dataoutputstream-how-do-i-define-the-endia

			DataOutputStream dout = new DataOutputStream(os);


			this.internalGraph.writeObject(dout);
		}
		Files.move(tempFile, file, REPLACE_EXISTING);
	}

	/**
	 * Read meta data for the {@link FeatureSpace}, number of nodes and edges
	 * 
	 * @param file
	 * @return
	 * @throws ClassNotFoundException
	 * @throws IOException
	 */
	public static DynamicExplorationGraph readFromFile(Path file) throws IOException {
		return new DynamicExplorationGraph(EvenRegularWeightedUndirectedGraph.readFromFile(file));
	}
	
	public static DynamicExplorationGraph readFromFile(Path file, String featureType) throws IOException {
		return new DynamicExplorationGraph(EvenRegularWeightedUndirectedGraph.readFromFile(file, featureType));
	}
}