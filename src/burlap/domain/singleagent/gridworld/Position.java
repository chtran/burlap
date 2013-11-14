package burlap.domain.singleagent.gridworld;

public class Position {
	public int x, y;
	
	public Position(int x, int y) {
		this.x = x;
		this.y = y;
	}
	
	@Override
    public boolean equals(Object obj) {
		boolean bothAreEqual = false;
		
        if (!(obj instanceof Position))
        	bothAreEqual = false;
        else if (this == obj)
        	bothAreEqual = true;
        else
        	bothAreEqual =    equal(x, ((Position) obj).x)
                           && equal(y, ((Position) obj).y);
        
        return bothAreEqual;
    }
	
    private boolean equal(Object o1, Object o2) {
        return o1 == null ? o2 == null : (o1 == o2 || o1.equals(o2));
    }
}