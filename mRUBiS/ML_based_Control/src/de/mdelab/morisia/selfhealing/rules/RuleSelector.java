package de.mdelab.morisia.selfhealing.rules;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.logging.Logger;

import com.fasterxml.jackson.core.JsonGenerationException;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import de.mdelab.morisia.comparch.AddReplica;
import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.CF1;
import de.mdelab.morisia.comparch.CF2;
import de.mdelab.morisia.comparch.CF3;
import de.mdelab.morisia.comparch.CF5;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ComponentType;
import de.mdelab.morisia.comparch.HwRedeployComponent;
import de.mdelab.morisia.comparch.Issue;
import de.mdelab.morisia.comparch.LwRedeployComponent;
import de.mdelab.morisia.comparch.RemoveReplica;
import de.mdelab.morisia.comparch.ReplaceComponent;
import de.mdelab.morisia.comparch.RestartComponent;
import de.mdelab.morisia.comparch.Rule;
import de.mdelab.morisia.comparch.Tenant;
import de.mdelab.morisia.selfhealing.Approaches;
import de.mdelab.morisia.selfhealing.Evaluation_ML;
import de.mdelab.morisia.selfhealing.LearningModel;
import de.mdelab.morisia.selfhealing.LearningQuality;
import de.mdelab.morisia.selfhealing.Utilityfunction;
import de.mdelab.morisia.selfhealing.rules.ComponentDependencies;
import mRUBiS.Observations.Observations;
import mRUBiS_Tasks.Input;
import mRUBiS_Tasks.Task_1;


public class RuleSelector {

	private static Random random = new Random();
	
    private static Path issueToRulesPath = Paths.get("issueToRulesMap.json"); // issues, affected components and associated rules;
    private static Path rulesToExecutePath= Paths.get("rulesToExecute.json"); // rules to execute on this run
    private static int run = 1;
    private static HashMap<String, HashMap<String, HashMap<String, String>>> globalState;
//    private static Boolean initialStateSent = false;
	

	private final static Logger LOGGER = Logger.getLogger(RuleSelector.class
			.getName());
	
	/**
	 * For the given issue, the method removes all rule markers except of one
	 * from the instance graph. The remaining rule marker is the selected rule
	 * to be executed in order to handle the given issue.
	 * 
	 * @param issue
	 *            the given issue.
	 */
	public static void selectRule(Issue issue) {
		LOGGER.info("Selecting rules for the issue " + issue);
		
		Utilityfunction UTILITY_FUNCTION = Utilityfunction.Optimal_Combined;
		
		LearningQuality quality=null;
		LearningModel model=null;
	
			
			
		if (Task_1.CURRENT_APPROACH==Approaches.Learning){
			System.out.print("\n >> Loading Input to select Actions");
			learningApproach(issue , UTILITY_FUNCTION );
			
		}
		else if (Task_1.CURRENT_APPROACH == Approaches.Udriven) {
			ourApproach(issue , UTILITY_FUNCTION , quality, model);
		}else if (Task_1.CURRENT_APPROACH == Approaches.RANDOM) {

			randomRuleSelection(issue);
			CostPredictor.predictCosts(issue);

			//UtilityIncreasePredictor.R_linearPredictUtility(issue);
			
			//UtilityIncreasePredictor.R_discontinousPredictUtility(issue);
			//UtilityIncreasePredictor.R_Nonlinear_SATURPredictUtility(issue);
		//combined
			UtilityIncreasePredictor.R_ALLbutRandomPredictUtility(issue);
		

		}
		
	}


	private static void learningApproach(Issue issue, Utilityfunction uTILITY_FUNCTION) {
		// TODO: get issueToRulesMap from UtilityIncreasePredictor, use getAffectedComponentStatus with it, put state into JSON for all issues
		
		
		// read utility increase (make sure that rules are available)
		HashMap<String, HashMap<String, HashMap<String, HashMap<String, Double>>>> shopToIssueMap;
		shopToIssueMap = UtilityIncreasePredictor.calculateCombinedUtilityIncrease(issue);
		updateGlobalState(shopToIssueMap);
		// TODO: We can really solve this a lot easier by just outputting a shop-component-issue triple, right? Or can this contain multiple shops' issues?
		
		Architecture architecture = issue.getAnnotations().getArchitecture();

//		sendNumberOfIssuesPerShopToPython(architecture);
//		sendCurrentIssueToPython(issue);
		
		getRuleFromPython(issue);
		Input.selectAction(issue);
		
	}
	
	
	private static void updateGlobalState(HashMap<String, HashMap<String, HashMap<String, HashMap<String, Double>>>> shopToIssueMap) {
		for (HashMap.Entry<String, HashMap<String, HashMap<String, HashMap<String, Double>>>> shop : shopToIssueMap.entrySet()) {
			String shopName = shop.getKey();
			HashMap<String, HashMap<String, HashMap<String, Double>>> issues = shop.getValue();
			for (HashMap.Entry<String, HashMap<String, HashMap<String, Double>>> issue: issues.entrySet()) {
				String issueName = issue.getKey();
				HashMap<String, HashMap<String, Double>> components = issue.getValue();
				for (HashMap.Entry<String, HashMap<String, Double>> component: components.entrySet()) {
					String componentName = component.getKey();
					HashMap<String, Double> rules = component.getValue();
					Double maxUtilityIncrease = 0.0;
					for (HashMap.Entry<String, Double> rule: rules.entrySet()) {
						maxUtilityIncrease = Math.max(maxUtilityIncrease, rule.getValue());
					}
					
					globalState.get(shopName).get(componentName).put("failure_name", issueName);
					// TODO: update utility?
				}
			}
		}
	}
	
	
	public static void sendFailProbabilityToPython() {
		String fromPython = "";
		System.out.println("Waiting for Python to send json...");
		while(true) { 
			fromPython = ChunkedSocketCommunicator.readln();
			if (fromPython.equals("fail_probability_finished")) {
				System.out.println("Finished writing fail probabilities.");
				break;
			}
			else {
				try {
					HashMap<String, ArrayList<String>> failProbabilityMessage = new HashMap<String, ArrayList<String>>();
					failProbabilityMessage = new ObjectMapper().readValue(fromPython, HashMap.class);
					
					String typeOfComponentToFix = failProbabilityMessage.get("type_of_component_to_fix").get(0);
					ArrayList<String> failingComponentTypes = failProbabilityMessage.get("failing_component_types");
					
					Double failProbability = ComponentDependencies.getFixFailProbability(typeOfComponentToFix, failingComponentTypes);
					
					HashMap<String, Double> failProbabilityOutput = new HashMap<String, Double>();					
					failProbabilityOutput.put("get_fail_probability", failProbability);
					
					String json = "";
					try {
						json = new ObjectMapper().writeValueAsString(failProbabilityOutput);
					} catch (JsonProcessingException e) {
						e.printStackTrace();
					}
					
					ChunkedSocketCommunicator.println(json);
				} catch (IOException e) {
					System.out.println("Did not receive valid json from Python:");
					System.out.println(fromPython);
				}
			}
		}
	}
	
	
	public static void getInitialState(Architecture MRUBIS) {
		ChunkedSocketCommunicator.waitForMessage("get_initial_state");
		HashMap<String, HashMap<String, HashMap<String, String>>> state;
		globalState = Observations.getInitialStates(MRUBIS);
		String json = "";
		try {
			json = new ObjectMapper().writeValueAsString(globalState);
			ChunkedSocketCommunicator.println(json);
		} catch (JsonProcessingException e) {
			e.printStackTrace();
		}	
	}
	
	
//	private static void sendNumberOfIssuesPerShopToPython(Architecture architecture) {
//		ChunkedSocketCommunicator.waitForMessage("get_number_of_issues_in_run");
//		globalState = Observations.getNumberOfIssuesPerShop(architecture);
//	
//		String json = "";
//		try {
//			json = new ObjectMapper().writeValueAsString(globalState);
//			ChunkedSocketCommunicator.println(json);
//		} catch (JsonProcessingException e) {
//			e.printStackTrace();
//		}	
//		ChunkedSocketCommunicator.println(json);		
//	}
	
	
//	private static void sendCurrentIssueToPython(Issue issue, shopToIssueMap) {
//		// Read json file generated in UtilityIncreasePredictor
//		ObjectMapper mapper = new ObjectMapper();
//		HashMap<String, HashMap<String, HashMap<String, HashMap<String, Double>>>> issueToRulesMapFromFile = null;
////		try {
////			issueToRulesMapFromFile = mapper.readValue(issueToRulesPath.toFile(), HashMap.class);
////		} catch (IOException e2) {
////			e2.printStackTrace();
////		}
//
//		// send current state to the python side
//		Architecture architecture = issue.getAnnotations().getArchitecture();
//		ChunkedSocketCommunicator.waitForMessage("get_issue");
//		String state = "not_available";
//		state = Observations.getAffectedComponentStatus(architecture, issueToRulesMapFromFile);
//		ChunkedSocketCommunicator.println(state);		
//	}
		
	private static void getRuleFromPython (Issue issue) {
		// Get the rules to execute from the python side
		System.out.println("Waiting for rules from Python side...");
		HashMap<String, HashMap<String, HashMap<String, String>>> rulesToExecute = ChunkedSocketCommunicator.readJSON(new HashMap<String, HashMap<String, HashMap<String, String>>>());
		try {
			ObjectMapper fromPythonMapper = new ObjectMapper();
			fromPythonMapper.writeValue(rulesToExecutePath.toFile(), rulesToExecute);
		} catch (IOException e) {
			System.out.println("Failed to write JSON to file:");
			System.out.println(rulesToExecute);
			e.printStackTrace();
		}
		
		ChunkedSocketCommunicator.println("rule_received");
		
		System.out.println("Rule received: " + rulesToExecute);
		
	}

	




	public static void addActualUtilityIncreaseToRule(Issue issue, Utilityfunction UTILITY_FUNCTION) {
	 if (UTILITY_FUNCTION==Utilityfunction.Combined)
			// calculate the increase using an analytical Combined utility function
			UtilityIncreasePredictor.calculateCombinedUtilityIncrease(issue);
		
	}
	
	/*
	 * OUR APPROACH ===========================================================
	 */

	private static void ourApproach(Issue issue, Utilityfunction UTILITY_FUNCTION ,  LearningQuality quality, LearningModel model) {

	
	if (UTILITY_FUNCTION==Utilityfunction.Optimal_Combined)
			// calculate the increase using an analytical Combined utility function
			UtilityIncreasePredictor.calculateCombinedUtilityIncrease(issue);

			
		
		 // select rules with the highest predicted utilityIncrease, that is,
		// remove all other rules from the instance graph
		selectBestRulesByUtilityIncrease(issue);
		// predict the costs as we require them for the evaluation to
		// estimate
		// the time of actually enacting an adaptation to the running
		// system.
		CostPredictor.predictCosts(issue);
		// check if more than one rule is available. All of these rules have
		// the
		// same predicted utilityIncrease.
		if (issue.getHandledBy().size() > 1) {
			// further filtering the rules based on the costs of each rule.
			selectBestRulesByCosts(issue);
		}

		// Theoretically, after selecting the rules with highest utility
		// increase and then selecting among those rules the ones with
		// lowest
		// costs, more than one rule could remain. Select an arbitrary one.
		if (issue.getHandledBy().size() > 1) {
			int numberOfRules = issue.getHandledBy().size();
			int ruleToUse = random.nextInt(numberOfRules);
			List<Rule> rulesToRemove = new LinkedList<>();
			for (int i = 0; i < numberOfRules; i++) {
				if (i != ruleToUse) {
					rulesToRemove.add(issue.getHandledBy().get(i));
				}
			}

			for (Rule rule : rulesToRemove) {
				LOGGER.info("Randomly (all remaining rules have the same utlityIncrease and costs) removing the rule "
						+ rule + " for issue " + issue);
				//EcoreUtil.delete(rule, true);
				rule.setAnnotations(null);
				rule.setHandles(null);
			}
		}

		// Now there is only one rule left that handles the given issue.
		if (issue.getHandledBy().size() > 1 || issue.getHandledBy().size() == 0) {
			throw new RuntimeException(issue.getHandledBy().size()
					+ " rules are assigned to the issue " + issue);
		} else {
			LOGGER.info("Selected rule " + issue.getHandledBy().get(0)
					+ " for issue " + issue);
			System.out.print("\n\n >>"+issue.getHandledBy().get(0).eClass().getName()+"  Selected for Execution");
			
		}
	}
	/*
	 * worst APPROACH ===========================================================
	 */


	/**
	 * Selects the best rules for the given issue based on the predicted utility
	 * increase of each rule. The other rules are removed from the instance
	 * graph.
	 * 
	 * @param issue
	 *            the given issue.
	 */
	private static void selectBestRulesByUtilityIncrease(Issue issue) {
		double maxUtilityIncrease = 0;
		List<Rule> rulesWithHighestUtilityIncrease = new LinkedList<Rule>();
		List<Rule> rulesWithLowerUtilityIncrease = new LinkedList<Rule>();
		
		for (Rule rule : issue.getHandledBy()) {
			double utilityIncrease = rule.getUtilityIncrease();
			//System.out.print("\n _________________Reading Rules  "+rule.eClass().getName());
			Component component=issue.getAffectedComponent();
		
		//	System.out.print("\n max U increase is "+rule.getUtilityIncrease());
			if (utilityIncrease < maxUtilityIncrease) {
				
				rulesWithLowerUtilityIncrease.add(rule);
			} else if (utilityIncrease == maxUtilityIncrease) {
			//	System.out.print("\n one rule added");
				rulesWithHighestUtilityIncrease.add(rule);
			} else if (utilityIncrease > maxUtilityIncrease) {
				maxUtilityIncrease = utilityIncrease;
				//System.out.print("\n better rule added");
				rulesWithLowerUtilityIncrease
				.addAll(rulesWithHighestUtilityIncrease);
				rulesWithHighestUtilityIncrease.clear();
				rulesWithHighestUtilityIncrease.add(rule);
			}
		}
	

		// remove rules that do not promise the best utility increase.
		for (Rule rule : rulesWithLowerUtilityIncrease) {
			LOGGER.info("UtilityIncrease-based removing rule " + rule
					+ " for issue " + issue);
			//EcoreUtil.delete(rule, true);
			rule.setAnnotations(null);
			rule.setHandles(null);
		}
		

	}

	/**
	 * Selects the best rules for the given issue based on the predicted costs
	 * of each rule. The other rules are removed from the instance graph.
	 * 
	 * @param issue
	 *            the given issue.
	 */
	private static void selectBestRulesByCosts(Issue issue) {
		double lowestCosts = Double.MAX_VALUE;
		List<Rule> rulesWithLowestCosts = new LinkedList<Rule>();
		List<Rule> rulesWithHigherCosts = new LinkedList<Rule>();
		for (Rule rule : issue.getHandledBy()) {
			double costs = rule.getCosts();
			if (costs > lowestCosts) {
				rulesWithHigherCosts.add(rule);
			} else if (costs == lowestCosts) {
				rulesWithLowestCosts.add(rule);
			} else if (costs < lowestCosts) {
				lowestCosts = costs;
				rulesWithHigherCosts.addAll(rulesWithLowestCosts);
				rulesWithLowestCosts.clear();
				rulesWithLowestCosts.add(rule);
			}
		}

		// remove rules that do have higher predicted costs.
		for (Rule rule : rulesWithHigherCosts) {
			LOGGER.info("Cost-based removing rule " + rule + " for issue "
					+ issue);
			//EcoreUtil.delete(rule, true);
			rule.setAnnotations(null);
			rule.setHandles(null);
		}

	}
	/**
	 * Selects the worst rules for the given issue based on the predicted utility
	 * increase of each rule. The other rules are removed from the instance
	 * graph.
	 * 
	 * @param issue
	 *            the given issue.
	 */
	private static void selectWorstRulesByUtilityIncrease(Issue issue) {
		double minUtilityIncrease = Double.MAX_VALUE;
		List<Rule> rulesWithLowestUtilityIncrease = new LinkedList<Rule>();
		List<Rule> rulesWithhigherUtilityIncrease = new LinkedList<Rule>();
		for (Rule rule : issue.getHandledBy()) {
			
			double utilityIncrease = rule.getUtilityIncrease();
		//	System.out.print("\n increase for rule   "+rule.eClass().getName()+"   " + utilityIncrease);
			if (utilityIncrease > minUtilityIncrease) {
				rulesWithhigherUtilityIncrease.add(rule);
			} else if (utilityIncrease == minUtilityIncrease) {
				rulesWithLowestUtilityIncrease.add(rule);
			} else if ((utilityIncrease < minUtilityIncrease)&(!(utilityIncrease==0))) {
				minUtilityIncrease = utilityIncrease;
				rulesWithhigherUtilityIncrease
				.addAll(rulesWithLowestUtilityIncrease);
				rulesWithLowestUtilityIncrease.clear();
				rulesWithLowestUtilityIncrease.add(rule);
			}
		}
		System.out.print("\n\n selected  " + rulesWithLowestUtilityIncrease+"\n ________________________________________________________________________________________\n");
		// remove rules that do not promise the best utility increase.
		for (Rule rule : rulesWithhigherUtilityIncrease) {
			LOGGER.info("UtilityIncrease-based removing rule " + rule
					+ " for issue " + issue);
			//EcoreUtil.delete(rule, true);
			rule.setAnnotations(null);
			rule.setHandles(null);
		}

	}

	/**
	 * Selects the worst rules for the given issue based on the predicted costs
	 * of each rule. The other rules are removed from the instance graph.
	 * 
	 * @param issue
	 *            the given issue.
	 */
	private static void selectWorstRulesByCosts(Issue issue) {
		double highestCosts = Double.MIN_VALUE;
		List<Rule> rulesWithLowerCosts = new LinkedList<Rule>();
		List<Rule> rulesWithHighestCosts = new LinkedList<Rule>();
		for (Rule rule : issue.getHandledBy()) {
			double costs = rule.getCosts();
			if (costs < highestCosts) {
				rulesWithLowerCosts.add(rule);
			} else if (costs == highestCosts) {
				rulesWithHighestCosts.add(rule);
			} else if (costs > highestCosts) {
				highestCosts = costs;
				rulesWithLowerCosts.addAll(rulesWithHighestCosts);
				rulesWithHighestCosts.clear();
				rulesWithHighestCosts.add(rule);
			}
		}
		System.out.print("\n\n selected by cost " + rulesWithHighestCosts+"\n ________________________________________________________________________________________\n");
		// remove rules that do have higher predicted costs.
		for (Rule rule : rulesWithLowerCosts) {
			LOGGER.info("Cost-based removing rule " + rule + " for issue "
					+ issue);
			//EcoreUtil.delete(rule, true);
			rule.setAnnotations(null);
			rule.setHandles(null);
		}

	}


	/*
	 * STATIC AND SEMI_STATIC APPROACHES
	 * ===========================================================
	 */

	private static void staticRuleSelection(Issue issue) {
		if (issue instanceof CF1) {
			CF1 cf1 = (CF1) issue;
			List<Rule> assignedRules = new LinkedList<Rule>();
			assignedRules.addAll(cf1.getHandledBy());
			for (Rule r : assignedRules) {
				if (!(r instanceof LwRedeployComponent)) {
					LOGGER.info("Statically removing rule " + r + " for issue "
							+ issue);
					//EcoreUtil.delete(r, true);
					r.setAnnotations(null);
					r.setHandles(null);
				}
			}
		} else if (issue instanceof CF2) {
			CF2 cf2 = (CF2) issue;
			List<Rule> assignedRules = new LinkedList<Rule>();
			assignedRules.addAll(cf2.getHandledBy());
			for (Rule r : assignedRules) {
				if (!(r instanceof RestartComponent)) {
					LOGGER.info("Statically removing rule " + r + " for issue "
							+ issue);
					//EcoreUtil.delete(r, true);
					r.setAnnotations(null);
					r.setHandles(null);
				}
			}
		} else if (issue instanceof CF3) {
			Component affectedComponent = issue.getAffectedComponent();
			ComponentType alternative = staticSelectAlternativeComponentType(affectedComponent);

			CF3 cf3 = (CF3) issue;
			List<Rule> assignedRules = new LinkedList<Rule>();
			assignedRules.addAll(cf3.getHandledBy());

			if (alternative != null) {
				for (Rule r : assignedRules) {
					if (r instanceof HwRedeployComponent) {
						LOGGER.info("Statically removing rule " + r
								+ " for issue " + issue);
						//EcoreUtil.delete(r, true);
						r.setAnnotations(null);
						r.setHandles(null);
					} else if (r instanceof ReplaceComponent) {
						ReplaceComponent replaceComponent = (ReplaceComponent) r;
						replaceComponent
						.setAlternativeComponentType(alternative);
					}
				}
			} else {
				for (Rule r : assignedRules) {
					if (r instanceof ReplaceComponent) {
						LOGGER.info("Statically removing rule " + r
								+ " for issue " + issue);
						//EcoreUtil.delete(r, true);
						r.setAnnotations(null);
						r.setHandles(null);
					}
				}
			}

		}

		// Now there is only one rule left that handles the given issue.
		if (issue.getHandledBy().size() > 1 || issue.getHandledBy().size() == 0) {
			throw new RuntimeException(issue.getHandledBy().size()
					+ " rules are assigned to the issue " + issue);
		} else {
			LOGGER.info("Selected rule " + issue.getHandledBy().get(0)
					+ " for issue " + issue);
		}
	}

	private static ComponentType staticSelectAlternativeComponentType(
			Component affectedComponent) {
		ComponentType affectedComponentType = affectedComponent.getType();
		ComponentType alternative = null;

		if (affectedComponentType.getName().equals("Authentication Service")) {
			for (ComponentType ct : affectedComponent.getTenant()
					.getArchitecture().getComponentTypes()) {
				if (ct.getName().equals("Google Authentication Service")) {
					alternative = ct;
					break;
				}
			}
		} else if (affectedComponentType.getName().equals(
				"Google Authentication Service")) {
			for (ComponentType ct : affectedComponent.getTenant()
					.getArchitecture().getComponentTypes()) {
				if (ct.getName().equals("Authentication Service")) {
					alternative = ct;
					break;
				}
			}
		}
		return alternative;
	}

	
	/**
	 * Selects the best rules for the given issue based on the predicted utility
	 * increase of each rule. The other rules are removed from the instance
	 * graph.
	 * 
	 * @param issue
	 *            the given issue.
	 */
	private static void solverselectBestRulesByUtilityIncrease(Issue issue) {

		List<Rule> rulesWithHighestUtilityIncrease = new LinkedList<Rule>();
		List<Rule> rulesWithLowerUtilityIncrease = new LinkedList<Rule>();
		double[] utilityIncreasevalues = new double[4];
		double[] selectedRulesByUtility = new double[4];


		// read utility values from all 4 types of rules and save in an array readable by Cplex solver
		for (Rule rule : issue.getHandledBy()) {
			if (rule instanceof RestartComponent) utilityIncreasevalues[0] = rule.getUtilityIncrease();
			else if  (rule instanceof LwRedeployComponent) utilityIncreasevalues[1] = rule.getUtilityIncrease();
			else if (rule instanceof HwRedeployComponent) utilityIncreasevalues[2] = rule.getUtilityIncrease();
			else if (rule instanceof ReplaceComponent) utilityIncreasevalues[3] = rule.getUtilityIncrease();



		}

		//solve





		//interpret the solver output 
		//"selectedRulesByUtility"; array of selected rules 
		//based on max utility increase

		for (Rule rule : issue.getHandledBy()) {

			if (rule instanceof RestartComponent)
				if (selectedRulesByUtility[0]== 0) {
					rulesWithLowerUtilityIncrease.add(rule);
				} else  {
					rulesWithHighestUtilityIncrease.add(rule);
				}

			else if  (rule instanceof LwRedeployComponent) 
				if (selectedRulesByUtility[1]== 0) {
					rulesWithLowerUtilityIncrease.add(rule);
				} else  {
					rulesWithHighestUtilityIncrease.add(rule);
				}
			else if (rule instanceof HwRedeployComponent)
				if (selectedRulesByUtility[2]== 0) {
					rulesWithLowerUtilityIncrease.add(rule);
				} else  {
					rulesWithHighestUtilityIncrease.add(rule);
				}
			else if (rule instanceof ReplaceComponent)
				if (selectedRulesByUtility[3]== 0) {
					rulesWithLowerUtilityIncrease.add(rule);
				} else  {
					rulesWithHighestUtilityIncrease.add(rule);
				}

		}

		// remove rules that do not promise the best utility increase.
		for (Rule rule : rulesWithLowerUtilityIncrease) {
			LOGGER.info("UtilityIncrease-based removing rule " + rule
					+ " for issue " + issue);
			//EcoreUtil.delete(rule, true);
			rule.setAnnotations(null);
			rule.setHandles(null);
		}

	}

	/**
	 * Selects the best rules for the given issue based on the predicted costs
	 * of each rule. The other rules are removed from the instance graph.
	 * 
	 * @param issue
	 *            the given issue.
	 */
	private static void solverselectBestRulesByCosts(Issue issue) {
		List<Rule> rulesWithLowestCosts = new LinkedList<Rule>();
		List<Rule> rulesWithHigherCosts = new LinkedList<Rule>();


		double[]costvalues = {Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE };
		double[] selectedRulesByCost = new double[4];


		// read cost values from all 4 types of rules and save in an array readable by Cplex solver
		for (Rule rule : issue.getHandledBy()) {
			if (rule instanceof RestartComponent) costvalues[0] = rule.getCosts();
			else if  (rule instanceof LwRedeployComponent) costvalues[1] = rule.getCosts();
			else if (rule instanceof HwRedeployComponent) costvalues[2] = rule.getCosts();
			else if (rule instanceof ReplaceComponent) costvalues[3] = rule.getCosts();;

		}



		//solve

	




		//interpret the solver output 
		//"selectedRulesByCost"; array of selected rules 
		//based on Min cost

		for (Rule rule : issue.getHandledBy()) {

			if (rule instanceof RestartComponent)
				if (selectedRulesByCost[0]== 0) {
					rulesWithHigherCosts.add(rule);
				} else  {
					rulesWithLowestCosts.add(rule);
				}


			if (rule instanceof LwRedeployComponent)
				if (selectedRulesByCost[1]== 0) {
					rulesWithHigherCosts.add(rule);
				} else  {
					rulesWithLowestCosts.add(rule);
				}


			if (rule instanceof HwRedeployComponent)
				if (selectedRulesByCost[2]== 0) {
					rulesWithHigherCosts.add(rule);
				} else  {
					rulesWithLowestCosts.add(rule);
				}


			if (rule instanceof ReplaceComponent)
				if (selectedRulesByCost[3]== 0) {
					rulesWithHigherCosts.add(rule);
				} else  {
					rulesWithLowestCosts.add(rule);
				}
		}




		// remove rules that do have higher predicted costs.
		for (Rule rule : rulesWithHigherCosts) {
			LOGGER.info("Cost-based removing rule " + rule + " for issue "
					+ issue);
			//EcoreUtil.delete(rule, true);
			rule.setAnnotations(null);
			rule.setHandles(null);
		}

	}

	/*
	 * Random ===========================================================
	 */

	private static void randomRuleSelection(Issue issue) {

		//for (int i = 0; i < issue.getHandledBy().size(); i++)
		//System.out.print("\n rules  " + issue.getHandledBy().get(i));
		if (issue instanceof CF5) 
		{
			if (issue.getAffectedComponent().getType().getPerformanceMax()-issue.getAffectedComponent().getPerformance()>0.1)

			{
				//System.out.print("\n ADD \n real perf  " + issue.getAffectedComponent().getPerformance() + "\n   Max perf  "+issue.getAffectedComponent().getType()
				//.getPerformanceMax());
				for (int i = 0; i < issue.getHandledBy().size(); i++) {
					Rule rule = issue.getHandledBy().get(i);
					if (!(rule instanceof AddReplica)) {
						rule.setAnnotations(null);
						rule.setHandles(null);
					}
				}
				//System.out.print("\n and the Add rule is  " + issue.getHandledBy().get(0));
			} else 
			{
				for (int i = 0; i < issue.getHandledBy().size(); i++) {
					Rule rule = issue.getHandledBy().get(i);
					if (!(rule instanceof RemoveReplica)) {
						rule.setAnnotations(null);
						rule.setHandles(null);
					}
				}
				//System.out.print("\n ^^^^^^^^^^^^^^^^^^^^^^^^^^ REMOVRE \n " + issue.getHandledBy().get(0));
			}

			/*for (Rule rule : rulesToRemove) {
				LOGGER.info("Randomly (all remaining rules have the same utlityIncrease and costs) removing the rule "
						+ rule + " for issue " + issue);
				//EcoreUtil.delete(rule, true);
				rule.setAnnotations(null);
				rule.setHandles(null);
			} */
		}

		// If the issues has more than one rule to handle it, Randomly select rules , that is, remove the rest
		else if (issue.getHandledBy().size() > 1) {


			int numberOfRules = issue.getHandledBy().size();
			int ruleToUse = random.nextInt(numberOfRules);
			/**/	//avoid assigning Replace rule for component types other than Authentication component
			if ( !issue.getAffectedComponent().getType().getName().equals(
					"Authentication Service") && issue.getHandledBy().get(ruleToUse) instanceof ReplaceComponent)
			{
				while (issue.getHandledBy().get(ruleToUse) instanceof ReplaceComponent)
				{
					ruleToUse = random.nextInt(numberOfRules);
				}
				//System.out.print("\n New rule to use: "+ issue.getHandledBy().get(ruleToUse));
			}/**/

			List<Rule> rulesToRemove = new LinkedList<>();
			for (int i = 0; i < numberOfRules; i++) {
				if (i != ruleToUse) {
					rulesToRemove.add(issue.getHandledBy().get(i));
				}
			}

			for (Rule rule : rulesToRemove) {
				LOGGER.info("Randomly (all remaining rules have the same utlityIncrease and costs) removing the rule "
						+ rule + " for issue " + issue);
				//EcoreUtil.delete(rule, true);
				rule.setAnnotations(null);
				rule.setHandles(null);
			}
		}


		// Now there is only one rule left that handles the given issue.
		if (issue.getHandledBy().size() > 1 || issue.getHandledBy().size() == 0) {
			throw new RuntimeException(issue.getHandledBy().size()
					+ " rules are assigned to the issue " + issue);
		} else {
			LOGGER.info("Selected rule " + issue.getHandledBy().get(0)
					+ " for issue " + issue);
		}

		//System.out.print("\n selected rule  "+ issue.getHandledBy().get(0)+"\n");
		//System.out.print("\n size of remaining rules for each issues  "+ issue.getHandledBy().size()+"\n");
	}



}//EOC
