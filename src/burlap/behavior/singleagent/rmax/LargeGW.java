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
import burlap.behavior.singleagent.Policy.ActionProb;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.rmax.SmallGW.LocRF;
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
 * Implementation of the large grid world described in 
 * Potential-based Shaping in Model-based Reinforcement Learning
 * Asmuth, Littman, Zinkov
 * Proceedings of the Twenty-Third AAAI Conference on Artificial Intelligence
 * 2008
 */
public class LargeGW {

	Domain						domain;
	GridWorldDomain				gwd;
	TerminalFunction			tf;
	RewardFunction				rf;
	RewardFunction				shapedRF;
	GridWorldPotential			gwp;
	int							m = 5; // rmax
	double						goalValue = 1.;
	double						rho = 0.8;
	double						discountFactor = 1.;
	double						stepCost = -0.001;
	boolean [][]				northWalls;
	boolean [][]				eastWalls;
	Position []					pitPos;
	Position []					goalPos;
	Position					initialAgentPos = new Position(4, 10);
	State 						initialState;
	DiscreteStateHashFactory	hashingFactory;
	StateParser					sp;
	int 						numberOfEpisodes = 1500;
	int 						maxEpisodeSize = 10000;
	Boolean						enablePositionReset = false;
	// Record data to outputPath
	Boolean						recordData = true;
	String outputPath = "/gpfs/main/home/oyakawa/Courses/2013-3-CS_2951F/Final_Project/output";

	public LargeGW() {
		gwd = new GridWorldDomain(15,15); // Column, Row (x,y)
		
		northWalls = new boolean [15][15];
		for (boolean [] row : northWalls)
			Arrays.fill(row, false);
		
		northWalls[1][0] = true;
		northWalls[6][0] = true;
		northWalls[8][0] = true;
		northWalls[11][0] = true;
		northWalls[9][1] = true;
		northWalls[13][1] = true;
		northWalls[2][2] = true;
		northWalls[5][2] = true;
		northWalls[8][2] = true;
		northWalls[3][3] = true;
		northWalls[7][3] = true;
		northWalls[10][3] = true;
		northWalls[12][3] = true;
		northWalls[13][3] = true;
		northWalls[4][4] = true;
		northWalls[11][4] = true;
		northWalls[1][5] = true;
		northWalls[3][5] = true;
		northWalls[6][5] = true;
		northWalls[7][5] = true;
		northWalls[8][5] = true;
		northWalls[11][6] = true;
		northWalls[2][7] = true;
		northWalls[3][7] = true;
		northWalls[6][7] = true;
		northWalls[8][7] = true;
		northWalls[4][8] = true;
		northWalls[6][8] = true;
		northWalls[7][8] = true;
		northWalls[11][8] = true;
		northWalls[2][9] = true;
		northWalls[8][9] = true;
		northWalls[11][9] = true;
		northWalls[6][10] = true;
		northWalls[7][10] = true;
		northWalls[6][11] = true;
		northWalls[10][11] = true;
		northWalls[13][11] = true;
		northWalls[1][12] = true;
		northWalls[3][12] = true;
		northWalls[7][12] = true;
		northWalls[9][12] = true;
		northWalls[4][13] = true;
		northWalls[9][13] = true;
		
		gwd.setNorthWalls(northWalls);
		
		eastWalls = new boolean [15][15];
		for (boolean [] row : eastWalls)
			Arrays.fill(row, false);
		
		eastWalls[0][1] = true;
		eastWalls[0][2] = true;
		eastWalls[0][3] = true;
		eastWalls[0][4] = true;
		eastWalls[0][5] = true;
		eastWalls[0][9] = true;
		eastWalls[0][12] = true;
		eastWalls[1][5] = true;
		eastWalls[1][6] = true;
		eastWalls[1][11] = true;
		eastWalls[2][1] = true;
		eastWalls[3][1] = true;
		eastWalls[3][2] = true;
		eastWalls[3][4] = true;
		eastWalls[3][6] = true;
		eastWalls[3][10] = true;
		eastWalls[3][11] = true;
		eastWalls[4][1] = true;
		eastWalls[4][2] = true;
		eastWalls[4][6] = true;
		eastWalls[4][8] = true;
		eastWalls[4][10] = true;
		eastWalls[5][1] = true;
		eastWalls[5][3] = true;
		eastWalls[5][14] = true;
		eastWalls[6][2] = true;
		eastWalls[6][6] = true;
		eastWalls[6][8] = true;
		eastWalls[6][13] = true;
		eastWalls[7][6] = true;
		eastWalls[7][7] = true;
		eastWalls[7][12] = true;
		eastWalls[9][0] = true;
		eastWalls[9][4] = true;
		eastWalls[9][6] = true;
		eastWalls[9][13] = true;
		eastWalls[10][1] = true;
		eastWalls[10][3] = true;
		eastWalls[10][6] = true;
		eastWalls[10][11] = true;
		eastWalls[10][13] = true;
		eastWalls[11][2] = true;
		eastWalls[11][12] = true;
		eastWalls[12][5] = true;
		eastWalls[12][6] = true;
		eastWalls[12][9] = true;
		eastWalls[12][10] = true;
		eastWalls[13][5] = true;
		eastWalls[13][6] = true;
		eastWalls[13][7] = true;
		eastWalls[13][9] = true;
		
		gwd.setEastWalls(eastWalls);
		
//		double [][] transitionDynamics = new double [][] {
//			{0.8, 0., 0.1, 0.1},
//			{0., 0.8, 0.1, 0.1},
//			{0.1, 0.1, 0.8, 0.},
//			{0.1, 0.1, 0., 0.8}
//		};
//		double [][] transitionDynamics = new double [][] {
//				{1., 0., 0., 0.},
//				{0., 1., 0., 0.},
//				{0., 0., 1., 0.},
//				{0., 0., 0., 1.}
//		};
		double rho2 = (1.-rho)/2.;
		double [][] transitionDynamics = new double [][] {
			{rho, 0., rho2, rho2, 0.},
			{0., rho, rho2, rho2, 0.},
			{rho2, rho2, rho, 0., 0.},
			{rho2, rho2, 0., rho, 0.},
			{0.,0.,0.,0.,1.}
		};
		gwd.setTransitionDynamics(transitionDynamics);
		gwd.populateDistance();
		domain = gwd.generateDomain();
		sp = new GridWorldStateParser(domain);
		
		// goals and pits
		goalPos = new Position [] {
			new Position(10,4),
		};
		pitPos = new Position [] {
			new Position(3,0),
			new Position(6,0),
			new Position(7,0),
			new Position(1,1),
			new Position(3,1),
			new Position(7,1),
			new Position(13,1),
			new Position(3,2),
			new Position(10,2),
			new Position(10,3),
			new Position(12,3),
			new Position(6,4),
			new Position(8,4),
			new Position(1,5),
			new Position(8,5),
			new Position(9,5),
			new Position(4,6),
			new Position(11,6),
			new Position(0,7),
			new Position(8,8),
			new Position(10,8),
			new Position(13,8),
			new Position(1,9),
			new Position(5,9),
			new Position(6,9),
			new Position(10,9),
			new Position(3,10),
			new Position(6,10),
			new Position(10,10),
			new Position(8,11),
			new Position(10,11),
			new Position(2,12),
			new Position(5,12),
			new Position(12,12),
			new Position(13,12),
			new Position(7,13),
			new Position(12,13),
			new Position(13,13),
			new Position(0,14),
			new Position(1,14),
			new Position(2,14),
			new Position(10,14)
			
		};
		
		tf = new LocTF(goalPos, pitPos);
		rf = new LocRF(goalPos, goalValue, pitPos, -1.0);
		gwp = new GridWorldPotential(gwd, goalPos[0], stepCost, rho, goalValue);
		initialState = GridWorldDomain.getOneAgentOneLocationState(domain);
		GridWorldDomain.setAgent(initialState, initialAgentPos.x, initialAgentPos.y);
		
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
		GridWorldDomain.setAgent(s, initialAgentPos.x, initialAgentPos.y);
		
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
		GridWorldDomain.setAgent(s, initialAgentPos.x, initialAgentPos.y);
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
		int numExperiments = 1;
		for (int ii = 0; ii < numExperiments; ii++) {
			System.out.println("Run " + ii + " of " + numExperiments);
			LargeGW myWorld = new LargeGW();
			myWorld.evaluatePolicy();
		}
	}

}
