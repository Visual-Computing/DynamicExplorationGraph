package com.vc.deg.designer;

import com.vc.deg.MemoryView;
import com.vc.deg.graph.WeightedUndirectedGraph;

/**
 * Verbessert und erweitert den Graphen
 * 
 * @author Neiko
 *
 */
public class EvenRegularGraphDesigner {
	
	protected WeightedUndirectedGraph graph;

	public EvenRegularGraphDesigner(WeightedUndirectedGraph graph) {
		this.graph = graph;
	}

	/**
	 * Add this new item to the graph
	 * 
	 * @param id
	 */
	public void add(int id, MemoryView data) {


	}
	
	public void extendGraph(int id, boolean randomPosition) {
		if(randomPosition)
			randomGraphExtension(id);
		
		
		// TODO anns entry point should be always the same node
		
	}
	
	protected void randomGraphExtension(int id) {
		// TODO use worst edges instead of random edge
	}

}
