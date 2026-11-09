import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

// Reads keys (one per line) and prints them in java.util.HashMap iteration order.
public class HashMapOrder {
    public static void main(String[] args) throws Exception {
        HashMap<String, Integer> map = new HashMap<>();
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
        String line;
        int i = 0;
        while ((line = in.readLine()) != null) {
            if (!line.isEmpty()) map.put(line, i++);
        }
        for (Map.Entry<String, Integer> e : map.entrySet()) System.out.println(e.getKey());
    }
}
