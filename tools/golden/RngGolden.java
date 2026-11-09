import com.megacrit.cardcrawl.random.Random;

// Prints reference outputs from the game's own RNG for Rust golden tests.
public class RngGolden {
    public static void main(String[] args) {
        long[] seeds = {0L, 1L, 42L, 123456789L, -987654321L};
        for (long seed : seeds) {
            Random r = new Random(seed);
            StringBuilder sb = new StringBuilder();
            sb.append(seed).append(" long=").append(r.randomLong());
            sb.append(" int99=").append(r.random(99));
            sb.append(" int5_10=").append(r.random(5, 10));
            sb.append(" bool=").append(r.randomBoolean());
            sb.append(" boolp=").append(r.randomBoolean(0.5625F));
            sb.append(" float=").append(Float.floatToIntBits(r.random()));
            sb.append(" longr=").append(r.random(1000000L));
            sb.append(" counter=").append(r.counter);
            java.util.Random jr = new java.util.Random(seed);
            sb.append(" jutil=").append(jr.nextInt(10)).append(",").append(jr.nextInt(1000)).append(",").append(jr.nextInt());
            System.out.println(sb);
        }
    }
}
