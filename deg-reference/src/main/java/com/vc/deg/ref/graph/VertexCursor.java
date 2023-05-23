package com.vc.deg.ref.graph;

import java.util.Iterator;

import com.vc.deg.FeatureVector;
import com.vc.deg.graph.NeighborConsumer;

public class VertexCursor implements com.vc.deg.graph.VertexCursor {

	protected final ArrayBasedWeightedUndirectedRegularGraph graph;
	protected final Iterator<VertexData> it;
	protected VertexData currentElement;
	
	public VertexCursor(ArrayBasedWeightedUndirectedRegularGraph graph) {
		this.graph = graph;
		this.it = graph.getVertices().iterator();
	}
	
	@Override
	public boolean moveNext() {
		if(it.hasNext()) {
			currentElement = it.next();
			return true;
		}
		
		return false;
	}
	
	@Override
	public int getVertexLabel() {
		return currentElement.getLabel();
	}

	@Override
	public FeatureVector getVertexFeature() {
		return currentElement.getFeature();
	}

	@Override
	public void forEachNeighbor(NeighborConsumer neighborConsumer) {
		currentElement.getEdges().forEach((int neighborId, float weight) -> 
			neighborConsumer.accept(graph.getVertexById(neighborId).getLabel(), weight)
		);
	}

}
