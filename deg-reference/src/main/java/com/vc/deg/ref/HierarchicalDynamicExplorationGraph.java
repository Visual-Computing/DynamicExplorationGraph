package com.vc.deg.ref;

import static java.nio.file.StandardCopyOption.REPLACE_EXISTING;
import static java.nio.file.StandardOpenOption.WRITE;
import static java.nio.file.StandardOpenOption.CREATE;
import static java.nio.file.StandardOpenOption.TRUNCATE_EXISTING;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.ref.hierarchy.HierarchicalGraphDesigner;

public class HierarchicalDynamicExplorationGraph implements com.vc.deg.HierarchicalDynamicExplorationGraph {

	protected final List<DynamicExplorationGraph> layers;
	protected final HierarchicalGraphDesigner designer;
	
	protected final FeatureSpace space;
	protected final int edgesPerVertex;
	protected final int topRankSize;
	
	/**
	 * Define how the nodes of the graph will be compared and how many edges each node has.
	 * 
	 * @param space
	 * @param edgesPerVertex
	 */
	public HierarchicalDynamicExplorationGraph(FeatureSpace space, int edgesPerVertex, int topRankSize) {
		this.layers = new ArrayList<>();
		this.space = space;
		this.edgesPerVertex = edgesPerVertex;
		this.topRankSize = topRankSize;
		this.designer = new HierarchicalGraphDesigner(layers, space, edgesPerVertex, topRankSize);
	}
	
	/**
	 * Used by the copy constructor and read file methods
	 * 
	 * @param layers
	 * @param space
	 * @param edgesPerVertex
	 * @param topRankSize
	 */
	private HierarchicalDynamicExplorationGraph(List<DynamicExplorationGraph> layers, FeatureSpace space, int edgesPerVertex, int topRankSize) {
		this.layers = layers;
		this.space = space;
		this.edgesPerVertex = edgesPerVertex;
		this.topRankSize = topRankSize;
		this.designer = new HierarchicalGraphDesigner(layers, space, edgesPerVertex, topRankSize);
	}
	
	@Override
	public HierarchicalGraphDesigner designer() {
		return designer;
	}

	/**
	 * Search the graph for the best vertices matching the query at the given hierarchy level.
	 * 
	 * @param query
	 * @param k
	 * @param eps Is similar to a search radius factor 0 means low and 1 means high radius to scan
	 * @return
	 */
	@Override
	public int[] search(FeatureVector query, int atLevel, int k, float eps) {
		return this.layers.get(atLevel).search(query, k, eps);
	}
	
	@Override
	public HierarchicalDynamicExplorationGraph copy() {
		final List<DynamicExplorationGraph> newLayers = new ArrayList<>();
		for (DynamicExplorationGraph dynamicExplorationGraph : this.layers) 
			newLayers.add(dynamicExplorationGraph.copy());
		return new HierarchicalDynamicExplorationGraph(newLayers, space, edgesPerVertex, topRankSize);
	}
	
	
	/**
	 * Stores each layer of the graph into a separate file.
	 * The target directory will be created if it does not exists.
	 * 
	 * @param targetDir path to a directory
	 * @throws ClassNotFoundException
	 * @throws IOException
	 */
	@Override
	public void writeToFile(Path targetDir) throws ClassNotFoundException, IOException {
		final Path tempDir = Paths.get(targetDir.getParent().toString(), "~$" + targetDir.getFileName().toString());
		Files.createDirectories(tempDir);
		Files.createDirectories(targetDir);

		// setup a list of graph files
		final List<String> graphFiles = new ArrayList<>();
		for (int i = 0; i < layers.size(); i++) 
			graphFiles.add("layer"+i+"."+space.getComponentType()+".deg");
		
		// create graph files	
		for (int i = 0; i < layers.size(); i++) {
			final DynamicExplorationGraph graph = layers.get(i);
			graph.writeToFile(tempDir.resolve(graphFiles.get(i)));
		}
		
		// delete old files
		Files.list(targetDir).forEach(f -> {
			try {
				Files.delete(f);
			} catch (IOException e) {
				e.printStackTrace();
			}
		});
		
		// move new files
		for (int i = 0; i < layers.size(); i++) {
			final Path sourceFile = tempDir.resolve(graphFiles.get(i));
			final Path targetFile = targetDir.resolve(graphFiles.get(i));
			Files.move(sourceFile, targetFile, REPLACE_EXISTING);			
		}
		
		// delete new directory
		Files.delete(tempDir);	
		
		// create hierarchical information file
		final List<String> info = new ArrayList<>();		
		info.add("featureType:"+space.getComponentType());
		info.add("featureSize:"+space.featureSize());
		info.add("featureDims:"+space.dims());
		info.add("featureMetric:"+space.metric());
		info.add("topRankSize:"+topRankSize);
		info.add("edgesPerVertex:"+edgesPerVertex);
		info.add("layerCount:"+layers.size());
		for (int i = 0; i < layers.size(); i++) 
			info.add(graphFiles.get(i));
		Files.write(targetDir.resolve("hierarchy.txt"), info, WRITE, CREATE, TRUNCATE_EXISTING);
	}
	
	/**
	 * Read the hierarchical graph from a directory containing a graph file for each layer.
	 * 
	 * @param targetDir
	 * @return
	 * @throws ClassNotFoundException
	 * @throws IOException
	 */
	public static HierarchicalDynamicExplorationGraph readFromFile(Path targetDir) throws IOException {
		final List<String> info = Files.readAllLines(targetDir.resolve("hierarchy.txt"));
		
		final String featureType = info.get(0).split(":")[1];
		final int featureSize = Integer.parseInt(info.get(1).split(":")[1]);
		final int dims = Integer.parseInt(info.get(2).split(":")[1]);
		final int metric = Integer.parseInt(info.get(3).split(":")[1]);
		
		// find the feature space specified in the file
		final FeatureSpace space = FeatureSpace.findFeatureSpace(featureType, metric, false);
		if(space == null)
			throw new UnsupportedOperationException("No feature space found for featureType="+featureType+", metric="+metric+" and isNative=false");
		if(featureSize != space.featureSize())
			throw new UnsupportedOperationException("The feature space for featureType="+featureType+", metric="+metric+" and isNative=false expects features with "+space.featureSize()+" bytes but the graph contains features with "+featureSize+" bytes.");
		if(dims != space.dims())
			throw new UnsupportedOperationException("The feature space for featureType="+featureType+", metric="+metric+" and isNative=false expects features with "+space.dims()+" dimensions but the graph contains features with "+dims+" dimensions.");
		
		
		final int topRankSize = Integer.parseInt(info.get(4).split(":")[1]);
		final int edgesPerVertex = Integer.parseInt(info.get(5).split(":")[1]);
		final int layerCount = Integer.parseInt(info.get(6).split(":")[1]);
		
		final List<DynamicExplorationGraph> newLayers = new ArrayList<>(layerCount);
		for (int i = 0; i < layerCount; i++) {
			final Path graphFile = targetDir.resolve(info.get(7 + i));
			newLayers.add(DynamicExplorationGraph.readFromFile(graphFile));	
		}	
		
		return new HierarchicalDynamicExplorationGraph(newLayers, space, edgesPerVertex, topRankSize);
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder("Layers: "+this.layers.size());
		sb.append('[');
		for (DynamicExplorationGraph deg : layers) {
			sb.append(deg.size());
			sb.append(',');
		}
		sb.append(']');
		return sb.toString();
	}
}