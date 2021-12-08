package de.mdelab.morisia.selfhealing.rules;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.Hashtable;

public class ComponentDependencies {
	
	private Hashtable<String, Hashtable<String, Double>> transitionMatrix = new Hashtable<String, Hashtable<String, Double>>();

	public ComponentDependencies() {
		this(Paths.get(System.getProperty("user.dir")).getParent().getParent().resolve("mrubis_controller").resolve("transition_matrix.csv"));
	}
	
	public ComponentDependencies(Path transitionMatrixPath) {
		try {
			BufferedReader br = new BufferedReader(new FileReader(transitionMatrixPath.toString()));
			String line = br.readLine();
			String[] keys = line.split(",");
			while ((line = br.readLine()) != null) {
				String[] values = line.split(",");
				Hashtable<String, Double> currentLine = new Hashtable<String, Double>();
				
				for (int j = 1; j < values.length; j++) {
					currentLine.put(keys[j], Double.parseDouble(values[j]));
				}				
				
				transitionMatrix.put(values[0], currentLine);		
			}
		} catch (FileNotFoundException e) {
			System.out.println("File " + transitionMatrixPath.toString() + " not found.");
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}	
	}
	
	public Double getFixFailProbability(String typeOfComponentToFix, ArrayList<String> failingComponentTypes) {
		Hashtable<String, Double> probabilities = transitionMatrix.get(typeOfComponentToFix);
		Double sum = 0.0;
		for (String failing : failingComponentTypes) {
			sum += probabilities.get(failing);
		}
		return sum;
	}
}
