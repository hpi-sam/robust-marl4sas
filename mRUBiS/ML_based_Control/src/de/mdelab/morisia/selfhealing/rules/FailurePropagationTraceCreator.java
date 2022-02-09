package de.mdelab.morisia.selfhealing.rules;

import java.util.Hashtable;
import java.util.LinkedList;
import java.util.List;

public class FailurePropagationTraceCreator {

	private static double propagationProbability = 0.5;

	private FailurePropagationTraceCreator() {

	}

	public static List<String> createTrace(String failingComponent) {
		List<String> trace = new LinkedList<String>();
		trace.add(failingComponent);

		String currentComponent = failingComponent;

		while (true) {
			double propagate = Math.random();
			if (propagate > propagationProbability) {
				break;
			}

			double choice = Math.random();			
			Hashtable<String, Double> possibleTransitions =  TransitionMatrixHandler.getTransitionMatrix().get(currentComponent);

			for (String component : possibleTransitions.keySet()) {
				if (possibleTransitions.get(component) > choice) {
					trace.add(component);
					currentComponent = component;
					break;
				}
				choice -= possibleTransitions.get(component);
			}
		}

		return trace;
	}
}