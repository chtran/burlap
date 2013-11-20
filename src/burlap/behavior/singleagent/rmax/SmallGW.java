package burlap.behavior.singleagent.rmax;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;

import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.Policy;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.shaping.potential.GridWorldPotential;
import burlap.behavior.singleagent.shaping.potential.PotentialShapedRF;
import burlap.behavior.statehashing.DiscreteStateHashFactory;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldStateParser;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.Position;
import burlap.oomdp.auxiliary.StateParser;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.ObjectInstance;
import burlap.oomdp.core.State;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.explorer.VisualExplorer;
import burlap.oomdp.visualizer.Visualizer;


/*
 * Implementation of the small grid world described in 
 * Potential-based Shaping in Model-based Reinforcement Learning
 * Asmuth, Littman, Zinkov
 * Proceedings of the Twenty-Third AAAI Conference on Artificial Intelligence
 * 2008
 */
public class SmallGW {

	Domain						domain;
	GridWorldDomain				gwd;
	TerminalFunction			tf;
	RewardFunction				rf;
	RewardFunction				shapedRF;
	GridWorldPotential			gwp;
	int							m = 5; // rmax
	double						goalValue = 5.;
	double						rho = 0.8;
	double						discountFactor = 1.;
	double						stepCost = -0.1;
	boolean [][]				northWalls;
	boolean [][]				eastWalls;
	Position []					pitPos;
	Position []					goalPos;
	Position					initialAgentPos = new Position(0, 5);
	State						initialState;
	DiscreteStateHashFactory	hashingFactory;
	StateParser					sp;
	int 						numberOfEpisodes = 2000;
	int 						maxEpisodeSize = 10000;
	Boolean						enablePositionReset = true;
	// Record data to outputPath
	Boolean						recordData = true;
	String outputPath = "/gpfs/main/home/oyakawa/Courses/2013-3-CS_2951F/Final_Project/output";
	
	public SmallGW() {
		gwd = new GridWorldDomain(4,6); // Column, Row (x,y)
				
		northWalls = new boolean [4][6];
		for (boolean [] row : northWalls)
			Arrays.fill(row, false);
		
		// Disable for q-learning -- suspect bug in paper
		northWalls[0][4] = true;
		
		gwd.setNorthWalls(northWalls);
		
		eastWalls = new boolean [4][6];
		for (boolean [] row : eastWalls)
			Arrays.fill(row, false);
		
		gwd.setEastWalls(eastWalls);
		
		gwd.setEnablePositionReset(enablePositionReset);
		
//		double [][] transitionDynamics = new double [][] {
//			{0.8, 0., 0.1, 0.1},
//			{0., 0.8, 0.1, 0.1},
//			{0.1, 0.1, 0.8, 0.},
//			{0.1, 0.1, 0., 0.8},
//		};
		double rho2 = (1.-rho)/2.;
		double [][] transitionDynamics = new double [][] {
			{rho, 0., rho2, rho2, 0.},
			{0., rho, rho2, rho2, 0.},
			{rho2, rho2, rho, 0., 0.},
			{rho2, rho2, 0., rho, 0.},
			{0.,0.,0.,0.,1.}
		};
//		double [][] transitionDynamics = new double [][] {
//				{rho, 0., rho2, rho2},
//				{0., rho, rho2, rho2},
//				{rho2, rho2, rho, 0.},
//				{rho2, rho2, 0., rho}
//		};
//		double [][] transitionDynamics = new double [][] {
//				{1., 0., 0., 0.},
//				{0., 1., 0., 0.},
//				{0., 0., 1., 0.},
//				{0., 0., 0., 1.},
//			};
		gwd.setTransitionDynamics(transitionDynamics);
		gwd.setInitialPosition(initialAgentPos.x, initialAgentPos.y);
		gwd.populateDistance();
		//gwd.printDistanceFrom(0, 5);
		domain = gwd.generateDomain();
		sp = new GridWorldStateParser(domain);
		
		// goals and pits
		goalPos = new Position [] {
			new Position(1,0),
		};
		pitPos = new Position [] {
			new Position(0,1),
			new Position(0,2),
			new Position(0,3),
			new Position(0,4),
			new Position(2,1),
			new Position(2,2),
			new Position(2,3),
			new Position(2,4),
		};
		
		tf = new LocTF(goalPos, pitPos);
		rf = new LocRF(goalPos, goalValue, pitPos, -1.0);
		gwp = new GridWorldPotential(gwd, goalPos[0], stepCost, rho, goalValue);
		shapedRF = new PotentialShapedRF(rf, gwp, discountFactor);
		initialState = GridWorldDomain.getOneAgentOneLocationState(domain);
		GridWorldDomain.setAgent(initialState, initialAgentPos.x, initialAgentPos.y);
		//GridWorldDomain.setLocation(initialState, 0, goalPos[0].x, goalPos[0].y);
		
		hashingFactory = new DiscreteStateHashFactory();
		hashingFactory.setAttributesForClass(GridWorldDomain.CLASSAGENT,
				domain.getObjectClass(GridWorldDomain.CLASSAGENT).attributeList);
		
		if(!outputPath.endsWith("/")) {
			outputPath = outputPath + "/";
		}
	}


	class LocTF implements TerminalFunction {
		Position [] tPos;
		
		public LocTF(Position [] goalPos, Position [] pitPos) {
			tPos = new Position[goalPos.length + pitPos.length];
			System.arraycopy(goalPos, 0, tPos, 0, goalPos.length);
			System.arraycopy(pitPos, 0, tPos, goalPos.length, pitPos.length);
		}
		
		@Override
		public boolean isTerminal(State s) {
			ObjectInstance agent = s.getObjectsOfTrueClass(GridWorldDomain.CLASSAGENT).get(0);
			Position aPos = new Position(
			    agent.getDiscValForAttribute(GridWorldDomain.ATTX),
			    agent.getDiscValForAttribute(GridWorldDomain.ATTY));
			
			boolean foundTerminalState = false;
			for (int p = 0; p < tPos.length; p++) {
				if (tPos[p].equals(aPos)) {
					foundTerminalState = true;
					break;
				}
			}
			return foundTerminalState;
		}
	}
	
	class LocRF implements RewardFunction {
		Position [] goalPos, pitPos;
		double goalReward, pitReward;
		
		public LocRF(Position [] goalPos, double goalReward,
				     Position [] pitPos, double pitReward) {
			this.goalPos = goalPos;
			this.pitPos = pitPos;
			this.goalReward = goalReward;
			this.pitReward = pitReward;
		}
		
		@Override
		public double reward(State s, GroundedAction a, State sprime) {
			ObjectInstance agent = sprime.getObjectsOfTrueClass(GridWorldDomain.CLASSAGENT).get(0);
			Position aPos = new Position(
				agent.getDiscValForAttribute(GridWorldDomain.ATTX),
				agent.getDiscValForAttribute(GridWorldDomain.ATTY));

			double r = stepCost;

			boolean foundGoal = false;
			for (int p = 0; p < goalPos.length  &&  !foundGoal; p++) {
				if (goalPos[p].equals(aPos)) {
					foundGoal = true;
					r = goalReward;
				}
			}
			for (int p = 0; p < pitPos.length  &&  !foundGoal; p++) {
				if (pitPos[p].equals(aPos)) {
					foundGoal = true;
					r = pitReward;
				}
			}
			
			return r;
		}
	}
	
	public void visualExplorer(){
		Visualizer v = GridWorldVisualizer.getVisualizer(domain, gwd.getMap(),
				gwd.getNorthWalls(), gwd.getEastWalls());
		
		State s = GridWorldDomain.getOneAgentNLocationState(domain, 0);
		GridWorldDomain.setAgent(s, 0, 5);
		
		VisualExplorer exp = new VisualExplorer(domain, v, s);
		exp.addKeyAction("w", GridWorldDomain.ACTIONNORTH);
		exp.addKeyAction("s", GridWorldDomain.ACTIONSOUTH);
		exp.addKeyAction("d", GridWorldDomain.ACTIONEAST);
		exp.addKeyAction("a", GridWorldDomain.ACTIONWEST);
		
		exp.initGUI();
	}
	
	public void evaluatePolicy() {
		//evaluateGoNorthPolicy();
		//evaluateQLearningPolicy();
		//evaluateQwithShapingLearningPolicy();
		evaluateRmaxLearningPolicy();
		//evaluateRmaxWithShapingLearningPolicy();
	}
	
	public void visualizeEpisode(String outputPath){
		Visualizer v = GridWorldVisualizer.getVisualizer(domain, gwd.getMap(), northWalls, eastWalls);
		EpisodeSequenceVisualizer evis = new EpisodeSequenceVisualizer(v, domain, sp, outputPath);
	}
	
	public void evaluateGoNorthPolicy() {
		Policy p = new Policy() {
			@Override
			public boolean isStochastic() {
				return false;
			}
			
			@Override
			public List<ActionProb> getActionDistributionForState(State s) {
				return this.getDeterministicPolicy(s);
			}
			
			@Override
			public GroundedAction getAction(State s) {
				return new GroundedAction(domain.getAction(GridWorldDomain.ACTIONNORTH), "");
			}
		};
		
		State s = GridWorldDomain.getOneAgentNLocationState(domain, 0);
		GridWorldDomain.setAgent(s, 0, 5);
		EpisodeAnalysis ea = p.evaluateBehavior(s, rf, tf, 1000);
		double pReturn = ea.getDiscountedReturn(discountFactor);
		System.out.println(pReturn);
	}
	
	public void evaluateQLearningPolicy() {
		QLearning qlplanner = new QLearning(domain, rf, tf,
				discountFactor, hashingFactory, 0., .02, maxEpisodeSize);
		qlplanner.setMaximumEpisodesForPlanning(numberOfEpisodes);
		qlplanner.setNumEpisodesToStore(numberOfEpisodes);
		qlplanner.planFromState(initialState);
		
		if (recordData) {
			outputEpisodeData(qlplanner.getAllStoredLearningEpisodes(),
							"ql_results.txt");
		}
	}
	
	public void evaluateQwithShapingLearningPolicy() {
		QLearning qlplanner = new QLearning(domain, shapedRF, tf,
				discountFactor, hashingFactory, 0., .02, maxEpisodeSize);
		qlplanner.setMaximumEpisodesForPlanning(numberOfEpisodes);
		qlplanner.setNumEpisodesToStore(numberOfEpisodes);
		qlplanner.planFromState(initialState);
		
		if (recordData) {
			outputEpisodeData(qlplanner.getAllStoredLearningEpisodes(),
							"qls_results.txt");
		}
	}
	
	public void evaluateRmaxLearningPolicy() {
		Rmax rmaxplanner = new Rmax(domain, rf, tf,
				discountFactor, hashingFactory, goalValue, maxEpisodeSize, m);
		rmaxplanner.setMaximumEpisodesForPlanning(numberOfEpisodes);
		rmaxplanner.setNumEpisodesToStore(numberOfEpisodes);
		rmaxplanner.planFromState(initialState);
		
		rmaxplanner.printRmaxDebug();
		
		if (recordData) {
			outputEpisodeData(rmaxplanner.getAllStoredLearningEpisodes(),
							"rmax_results.txt");
		}
	}
	
	public void evaluateRmaxWithShapingLearningPolicy() {
		Rmax rmaxplanner = new Rmax(domain, shapedRF, tf,
				discountFactor, hashingFactory, goalValue, maxEpisodeSize, m);
		rmaxplanner.setMaximumEpisodesForPlanning(numberOfEpisodes);
		rmaxplanner.setNumEpisodesToStore(numberOfEpisodes);
		rmaxplanner.planFromState(initialState);
		
		rmaxplanner.printRmaxDebug();
		
		if (recordData) {
			outputEpisodeData(rmaxplanner.getAllStoredLearningEpisodes(),
							"rmaxs_results.txt");
		}
	}
	
	public void outputEpisodeData(List<EpisodeAnalysis> episodes, String filename) {
		try {
			PrintWriter pw = new PrintWriter(
					new BufferedWriter(
							new FileWriter(outputPath + filename, true)));
			for (EpisodeAnalysis ea: episodes) {
				pw.println(String.valueOf(ea.getDiscountedReturn(discountFactor)));
			}
			pw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		//SmallGW myWorld = new SmallGW();	
		//myWorld.visualExplorer();
		int numExperiments = 40;
		for (int ii = 0; ii < numExperiments; ii++) {
			System.out.println("Run " + ii + " of " + numExperiments);
			SmallGW myWorld = new SmallGW();
			myWorld.evaluatePolicy();
		}
	}

}
