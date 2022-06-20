package com.vc.deg.ref.navigation;

import com.vc.deg.GraphNavigator;
import com.vc.deg.graph.WeightedUndirectedGraph;
import com.vc.deg.ref.graph.MapBasedWeightedUndirectedGraph;

/**
 * Erzeugt 2D projektionen von Subgraphen und übernimmt die Navigation davon.
 * Stellt allgemeine Suchfunktionen bereits
 * 
 * @author Neiko
 *
 */
public class MapBasedGraphNavigator implements GraphNavigator {

	protected MapBasedWeightedUndirectedGraph graph;

	public MapBasedGraphNavigator(MapBasedWeightedUndirectedGraph graph) {
		this.graph = graph;
	}

}
