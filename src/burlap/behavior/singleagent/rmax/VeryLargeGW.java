package burlap.behavior.singleagent.rmax;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.Policy;
import burlap.behavior.singleagent.Policy.ActionProb;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.rmax.SmallGW.LocRF;
import burlap.behavior.singleagent.rmax.SmallGW.LocTF;
import burlap.behavior.singleagent.shaping.potential.GridWorldPotential;
import burlap.behavior.singleagent.shaping.potential.PotentialShapedRF;
import burlap.behavior.statehashing.DiscreteStateHashFactory;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldStateParser;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.Position;
import burlap.oomdp.auxiliary.StateParser;
import burlap.oomdp.core.ObjectInstance;
import burlap.oomdp.core.State;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.SADomain;
import burlap.oomdp.singleagent.common.VisualActionObserver;
import burlap.oomdp.singleagent.explorer.VisualExplorer;
import burlap.oomdp.visualizer.Visualizer;

public class VeryLargeGW {

	SADomain					domain;
	GridWorldDomain				gwd;
	TerminalFunction			tf;
	RewardFunction				rf;
	RewardFunction				shapedRF;
	GridWorldPotential			gwp;
	int							m = 5; // rmax
	double						initial_temp = 100.0; // artdp
	double						min_temp = 0.01;
	double						temp_decay_constant = 0.996;
	double						goalValue = 1.;
	double						rho = 0.8;
	double						discountFactor = 1.;
	double						stepCost = -0.01;
	boolean [][]				northWalls;
	boolean [][]				eastWalls;
	Position []					pitPos;
	Position []					goalPos;
	Position					initialAgentPos = new Position(7, 17);
	State						initialState;
	DiscreteStateHashFactory	hashingFactory;
	StateParser					sp;
	int 						numberOfEpisodes = 1500;
	int 						maxEpisodeSize = 30000;
	Boolean						enablePositionReset = false;
	Boolean						visualizeEpisodes = false;
	// Record data to outputPath
	Boolean						recordData = true;
	String outputPath = "/gpfs/main/home/oyakawa/Courses/2013-3-CS_2951F/Final_Project/output";
	
	public VeryLargeGW() {
		gwd = new GridWorldDomain(25,25); // Column, Row (x,y)
		
		northWalls = new boolean [25][25];
		for (boolean [] row : northWalls)
			Arrays.fill(row, false);
		
		northWalls[1][10] = true;
		northWalls[1][15] = true;
		northWalls[1][22] = true;
		northWalls[2][4] = true;
		northWalls[2][5] = true;
		northWalls[2][12] = true;
		northWalls[2][4] = true;
		northWalls[2][17] = true;
		northWalls[2][19] = true;
		northWalls[3][1] = true;
		northWalls[3][7] = true;
		northWalls[3][7] = true;
		northWalls[3][13] = true;
		northWalls[3][15] = true;
		northWalls[3][17] = true;
		northWalls[3][7] = true;
		northWalls[3][22] = true;
		northWalls[4][9] = true;
		northWalls[4][14] = true;
		northWalls[4][18] = true;
		northWalls[4][9] = true;
		northWalls[4][23] = true;
		northWalls[5][3] = true;
		northWalls[5][5] = true;
		northWalls[5][12] = true;
		northWalls[6][0] = true;
		northWalls[6][8] = true;
		northWalls[6][10] = true;
		northWalls[6][0] = true;
		northWalls[6][15] = true;
		northWalls[6][17] = true;
		northWalls[6][18] = true;
		northWalls[6][20] = true;
		northWalls[6][21] = true;
		northWalls[7][4] = true;
		northWalls[7][5] = true;
		northWalls[7][7] = true;
		northWalls[7][13] = true;
		northWalls[7][15] = true;
		northWalls[7][15] = true;
		northWalls[7][18] = true;
		northWalls[7][20] = true;
		northWalls[7][22] = true;
		northWalls[8][0] = true;
		northWalls[8][1] = true;
		northWalls[8][0] = true;
		northWalls[8][5] = true;
		northWalls[8][10] = true;
		northWalls[8][12] = true;
		northWalls[8][15] = true;
		northWalls[8][17] = true;
		northWalls[8][19] = true;
		northWalls[9][4] = true;
		northWalls[9][6] = true;
		northWalls[9][11] = true;
		northWalls[9][22] = true;
		northWalls[9][23] = true;
		northWalls[10][0] = true;
		northWalls[10][1] = true;
		northWalls[10][13] = true;
		northWalls[10][21] = true;
		northWalls[11][3] = true;
		northWalls[11][10] = true;
		northWalls[11][14] = true;
		northWalls[11][16] = true;
		northWalls[11][18] = true;
		northWalls[11][19] = true;
		northWalls[12][1] = true;
		northWalls[12][8] = true;
		northWalls[12][9] = true;
		northWalls[12][13] = true;
		northWalls[13][2] = true;
		northWalls[13][4] = true;
		northWalls[13][5] = true;
		northWalls[13][11] = true;
		northWalls[13][13] = true;
		northWalls[13][21] = true;
		northWalls[14][1] = true;
		northWalls[14][5] = true;
		northWalls[14][6] = true;
		northWalls[14][6] = true;
		northWalls[14][20] = true;
		northWalls[15][5] = true;
		northWalls[16][1] = true;
		northWalls[16][8] = true;
		northWalls[16][17] = true;
		northWalls[17][1] = true;
		northWalls[17][3] = true;
		northWalls[17][16] = true;
		northWalls[17][18] = true;
		northWalls[18][4] = true;
		northWalls[18][7] = true;
		northWalls[18][13] = true;
		northWalls[18][15] = true;
		northWalls[18][22] = true;
		northWalls[19][5] = true;
		northWalls[19][14] = true;
		northWalls[19][15] = true;
		northWalls[19][17] = true;
		northWalls[19][18] = true;
		northWalls[19][19] = true;
		northWalls[19][21] = true;
		northWalls[20][3] = true;
		northWalls[20][8] = true;
		northWalls[20][10] = true;
		northWalls[20][11] = true;
		northWalls[20][20] = true;
		northWalls[20][23] = true;
		northWalls[21][9] = true;
		northWalls[21][18] = true;
		northWalls[21][19] = true;
		northWalls[22][3] = true;
		northWalls[22][13] = true;
		northWalls[23][10] = true;

		
		
		gwd.setNorthWalls(northWalls);
		
		eastWalls = new boolean [25][25];
		for (boolean [] row : eastWalls)
			Arrays.fill(row, false);
		
		eastWalls[0][1] = true;
		eastWalls[0][2] = true;
		eastWalls[0][15] = true;
		eastWalls[2][0] = true;
		eastWalls[2][4] = true;
		eastWalls[2][5] = true;
		eastWalls[2][6] = true;
		eastWalls[2][0] = true;
		eastWalls[2][17] = true;
		eastWalls[2][22] = true;
		eastWalls[3][1] = true;
		eastWalls[3][11] = true;
		eastWalls[3][11] = true;
		eastWalls[3][15] = true;
		eastWalls[3][16] = true;
		eastWalls[3][18] = true;
		eastWalls[4][3] = true;
		eastWalls[4][4] = true;
		eastWalls[4][3] = true;
		eastWalls[4][12] = true;
		eastWalls[4][21] = true;
		eastWalls[5][5] = true;
		eastWalls[5][5] = true;
		eastWalls[5][14] = true;
		eastWalls[5][14] = true;
		eastWalls[5][19] = true;
		eastWalls[5][24] = true;
		eastWalls[6][0] = true;
		eastWalls[6][2] = true;
		eastWalls[6][4] = true;
		eastWalls[6][6] = true;
		eastWalls[6][10] = true;
		eastWalls[6][21] = true;
		eastWalls[7][1] = true;
		eastWalls[7][5] = true;
		eastWalls[7][8] = true;
		eastWalls[7][11] = true;
		eastWalls[7][20] = true;
		eastWalls[7][23] = true;
		eastWalls[8][10] = true;
		eastWalls[8][12] = true;
		eastWalls[8][15] = true;
		eastWalls[8][20] = true;
		eastWalls[9][7] = true;
		eastWalls[9][9] = true;
		eastWalls[9][15] = true;
		eastWalls[9][16] = true;
		eastWalls[9][22] = true;
		eastWalls[10][2] = true;
		eastWalls[10][10] = true;
		eastWalls[10][12] = true;
		eastWalls[10][12] = true;
		eastWalls[10][14] = true;
		eastWalls[10][15] = true;
		eastWalls[10][18] = true;
		eastWalls[10][23] = true;
		eastWalls[11][0] = true;
		eastWalls[11][5] = true;
		eastWalls[11][13] = true;
		eastWalls[11][16] = true;
		eastWalls[11][23] = true;
		eastWalls[12][3] = true;
		eastWalls[12][10] = true;
		eastWalls[12][12] = true;
		eastWalls[12][16] = true;
		eastWalls[12][18] = true;
		eastWalls[13][11] = true;
		eastWalls[13][14] = true;
		eastWalls[13][21] = true;
		eastWalls[13][22] = true;
		eastWalls[14][16] = true;
		eastWalls[14][17] = true;
		eastWalls[15][0] = true;
		eastWalls[15][1] = true;
		eastWalls[15][2] = true;
		eastWalls[15][5] = true;
		eastWalls[15][11] = true;
		eastWalls[15][12] = true;
		eastWalls[15][16] = true;
		eastWalls[16][11] = true;
		eastWalls[16][12] = true;
		eastWalls[16][13] = true;
		eastWalls[16][20] = true;
		eastWalls[17][9] = true;
		eastWalls[17][11] = true;
		eastWalls[17][16] = true;
		eastWalls[17][19] = true;
		eastWalls[18][11] = true;
		eastWalls[18][12] = true;
		eastWalls[18][16] = true;
		eastWalls[18][19] = true;
		eastWalls[18][23] = true;
		eastWalls[19][1] = true;
		eastWalls[19][2] = true;
		eastWalls[19][3] = true;
		eastWalls[19][14] = true;
		eastWalls[19][24] = true;
		eastWalls[20][6] = true;
		eastWalls[20][8] = true;
		eastWalls[20][16] = true;
		eastWalls[20][17] = true;
		eastWalls[21][5] = true;
		eastWalls[21][13] = true;
		eastWalls[21][15] = true;
		eastWalls[21][16] = true;
		eastWalls[21][21] = true;
		eastWalls[22][7] = true;
		eastWalls[22][15] = true;
		eastWalls[22][21] = true;
		eastWalls[23][4] = true;
		eastWalls[23][8] = true;
		eastWalls[23][9] = true;
		eastWalls[23][10] = true;
		eastWalls[23][11] = true;
		eastWalls[23][12] = true;
		eastWalls[23][22] = true;



		
		
		gwd.setEastWalls(eastWalls);
		
		gwd.setEnablePositionReset(enablePositionReset);
		
		double rho2 = (1.-rho)/2.;
		double [][] transitionDynamics;
		if (enablePositionReset) {
			transitionDynamics = new double [][] {
					{rho, 0., rho2, rho2, 0.},
					{0., rho, rho2, rho2, 0.},
					{rho2, rho2, rho, 0., 0.},
					{rho2, rho2, 0., rho, 0.},
					{0.,0.,0.,0.,1.}};
		} else {
			transitionDynamics = new double [][] {
					{rho, 0., rho2, rho2},
					{0., rho, rho2, rho2},
					{rho2, rho2, rho, 0.},
					{rho2, rho2, 0., rho}};
		}

		gwd.setTransitionDynamics(transitionDynamics);
		gwd.setInitialPosition(initialAgentPos.x, initialAgentPos.y);
		gwd.populateDistance();
		domain = (SADomain) gwd.generateDomain();
		sp = new GridWorldStateParser(domain);
		
		// goals and pits
		goalPos = new Position [] {
			new Position(17,8),
		};
		pitPos = new Position [] {
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
				this.goalPos[0], this.pitPos, gwd.getNorthWalls(), gwd.getEastWalls());
		
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
		if (visualizeEpisodes) {
			GridWorldDomain.setLocation(initialState, 0, goalPos[0].x, goalPos[0].y);
			VisualActionObserver observer = new VisualActionObserver(domain,
					GridWorldVisualizer.getVisualizer(domain, gwd.getMap(),
							this.pitPos, this.northWalls, this.eastWalls));
			this.domain.setActionObserverForAllAction(observer);
			observer.initGUI();
		}
		//evaluateGoNorthPolicy();
		//evaluateQLearningPolicy();
		//evaluateQwithShapingLearningPolicy();
		//evaluateRmaxLearningPolicy();
		evaluateRmaxWithShapingLearningPolicy();
		//evaluateARTDPLearningPolicy();
		//evaluateARTDPWithHeuristicLearningPolicy();
	}
	
	public void visualizeEpisode(String outputPath){
		Visualizer v = GridWorldVisualizer.getVisualizer(domain, gwd.getMap(), this.pitPos, northWalls, eastWalls);
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

			@Override
			public boolean isDefinedFor(State s) {
				return true;
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
		LinkedList <Double> finalRewards = new LinkedList <Double> ();
		Rmax rmaxplanner = new Rmax(domain, rf, tf,
				discountFactor, hashingFactory, goalValue, maxEpisodeSize, m);
		rmaxplanner.setFinalRewardsList(finalRewards);
		rmaxplanner.setMaximumEpisodesForPlanning(numberOfEpisodes);
		//rmaxplanner.setNumEpisodesToStore(numberOfEpisodes);
		rmaxplanner.planFromState(initialState);
		
		rmaxplanner.printRmaxDebug();
		
		if (recordData) {
			outputEpisodeData(finalRewards,
							"rmax_results.txt");
		}
	}
	
	public void evaluateRmaxWithShapingLearningPolicy() {
		LinkedList <Double> finalRewards = new LinkedList <Double> ();
		RmaxWithShaping rmaxplanner = new RmaxWithShaping(domain, shapedRF, tf,
				discountFactor, hashingFactory, goalValue, maxEpisodeSize, m);
		rmaxplanner.setFinalRewardsList(finalRewards);
		rmaxplanner.setGridWorldPotential(this.gwp);
		rmaxplanner.setMaximumEpisodesForPlanning(numberOfEpisodes);
		//rmaxplanner.setNumEpisodesToStore(numberOfEpisodes);
		rmaxplanner.planFromState(initialState);
		
		rmaxplanner.printRmaxDebug();
		
		if (recordData) {
			outputEpisodeData(finalRewards,
							"rmaxs_results.txt");
		}
	}
	
	public void evaluateARTDPLearningPolicy() {
		LinkedList <Double> finalRewards = new LinkedList <Double> ();
		ARTDP artdpplanner = new ARTDP(domain, rf, tf,
				discountFactor, hashingFactory, goalValue, maxEpisodeSize, 1,
				initial_temp, min_temp, temp_decay_constant);
		artdpplanner.setFinalRewardsList(finalRewards);
		artdpplanner.setMaximumEpisodesForPlanning(numberOfEpisodes);
		//artdpplanner.setNumEpisodesToStore(numberOfEpisodes);
		artdpplanner.planFromState(initialState);
		
		artdpplanner.printRmaxDebug();
		
		if (recordData) {
			outputEpisodeData(finalRewards,
							"artdp_results.txt");
		}
	}
	
	public void evaluateARTDPWithHeuristicLearningPolicy() {
		LinkedList <Double> finalRewards = new LinkedList <Double> ();
		ARTDPHeuristic artdpplanner = new ARTDPHeuristic(domain, rf, tf,
				discountFactor, hashingFactory, goalValue, maxEpisodeSize, 1,
				initial_temp, min_temp, temp_decay_constant);
		artdpplanner.setFinalRewardsList(finalRewards);
		artdpplanner.setGridWorldPotential(this.gwp);
		artdpplanner.setMaximumEpisodesForPlanning(numberOfEpisodes);
		//artdpplanner.setNumEpisodesToStore(numberOfEpisodes);
		artdpplanner.planFromState(initialState);
		
		artdpplanner.printRmaxDebug();
		
		if (recordData) {
			outputEpisodeData(finalRewards,
							"artdph_results.txt");
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
	
	public void outputEpisodeData(LinkedList <Double> finalRewards, String filename) {
		try {
			PrintWriter pw = new PrintWriter(
					new BufferedWriter(
							new FileWriter(outputPath + filename, true)));
			for (double r: finalRewards) {
				pw.println(String.valueOf(r));
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
		VeryLargeGW myWorld = new VeryLargeGW();	
		myWorld.visualExplorer();
//		int numExperiments = 30;
//		for (int ii = 1; ii <= numExperiments; ii++) {
//			System.out.println("Run " + ii + " of " + numExperiments);
//			VeryLargeGW myWorld = new VeryLargeGW();
//			myWorld.evaluatePolicy();
//		}
	}

}
