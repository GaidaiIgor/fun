import java.io.FileInputStream;
import java.util.Map;
import java.util.Scanner;
import java.util.TreeMap;

public class Main
{
    public static int get_interval_capacity(int length, int ship_size)
    {
        return (length + 1) / (ship_size + 1);
    }

    public static void main(String[] args)
    {
        try
        {
            //FileInputStream input_file = new FileInputStream("input");
            //System.setIn(input_file);
            Scanner scanner = new Scanner(System.in);
            int n = scanner.nextInt();
            int k = scanner.nextInt();
            int a = scanner.nextInt();
            int m = scanner.nextInt();
//            int[] moves = new int[m];
//            for (int i = 0; i < m; ++i)
//            {
//                moves[i] = scanner.nextInt();
//            }

            TreeMap<Integer, Integer> intervals = new TreeMap<>();
            intervals.put(n + 1, n);
            int total_capacity = get_interval_capacity(n, a);
            boolean move_found = false;
            for (int i = 0; i < m; ++i)
            {
                int next_move = scanner.nextInt();
                Map.Entry<Integer, Integer> splitted_interval = intervals.ceilingEntry(next_move);
                int splitted_interval_capacity = get_interval_capacity(splitted_interval.getValue(), a);

                int left_interval_length = splitted_interval.getValue() - (splitted_interval.getKey() - next_move);
                int left_interval_capacity = get_interval_capacity(left_interval_length, a);
                intervals.put(next_move, left_interval_length);

                int right_interval_length = splitted_interval.getKey() - next_move - 1;
                int right_interval_capacity = get_interval_capacity(right_interval_length, a);
                intervals.replace(splitted_interval.getKey(), right_interval_length);

                total_capacity = total_capacity - splitted_interval_capacity + left_interval_capacity + right_interval_capacity;
                if (total_capacity < k)
                {
                    System.out.println(i + 1);
                    move_found = true;
                    break;
                }
            }
            if (!move_found)
            {
                System.out.println("-1");
            }
        }
        catch (Exception e)
        {
            System.out.println("error occurred");
        }
    }
}
