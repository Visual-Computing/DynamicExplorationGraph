package com.vc.deg.ref;

import java.io.IOException;
import java.nio.file.Path;

import com.vc.deg.FeatureSpace;

public class GraphFactory implements com.vc.deg.GraphFactory {

	@Override
	public DynamicExplorationGraph newGraph(FeatureSpace space, int edgesPerNode) {
		return new DynamicExplorationGraph(space, edgesPerNode);
	}

	@Override
	public DynamicExplorationGraph loadGraph(Path file) throws IOException {
		return DynamicExplorationGraph.readFromFile(file);
	}
	
	@Override
	public DynamicExplorationGraph loadGraph(Path file, String componentType) throws IOException {
		return DynamicExplorationGraph.readFromFile(file, componentType);
	}

}

