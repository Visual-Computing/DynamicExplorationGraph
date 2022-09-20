package com.vc.deg.ref.search;

import com.vc.deg.SearchResult.SearchEntry;
import com.vc.deg.graph.VertexView;

/**
 * TODO remove vertexId and nodeData
 * TODO move to common
 * 
 * @author Neiko
 *
 */
public class ObjectDistance implements SearchEntry {
	
	protected final int nodeId;
	protected final VertexView nodeData;
	protected final float distance;
	
	public ObjectDistance(int nodeId, VertexView nodeData, float distance) {
		this.nodeId = nodeId;
		this.nodeData = nodeData;
		this.distance = distance;
	}
	
	public int getVertexId() {
		return nodeId;
	}

	@Override
	public int getLabel() {
		return nodeData.getId();
	}

	@Override
	public float getDistance() {
		return distance;
	}
	
	@Override
	public String toString() {
		return "nodeId:"+nodeId+", nodeData:"+nodeData.getId()+", distance:"+distance;
	}
}