package com.vc.deg.impl.navigation;

import com.vc.deg.GraphNavigator;
import com.vc.deg.impl.graph.EvenRegularWeightedUndirectedGraph;

/**
 * Erzeugt 2D projektionen von Subgraphen und übernimmt die Navigation davon.
 * Stellt allgemeine Suchfunktionen bereits
 * 
 * @author Neiko
 *
 */
public class EvenRegularGraphNavigator implements GraphNavigator {

	protected EvenRegularWeightedUndirectedGraph graph;

	public EvenRegularGraphNavigator(EvenRegularWeightedUndirectedGraph graph) {
		this.graph = graph;
	}

}
