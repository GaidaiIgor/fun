import java.io.FileInputStream;
import java.util.*;

class Defibrillator
{
    public String name;
    public double longitude;
    public double latitude;

    public Defibrillator(String name, double longitude, double latitude)
    {
        this.name = name;
        this.longitude = longitude;
        this.latitude = latitude;
    }
}

class Solution
{
    public static double distance(double longitude_a, double latitude_a, double longitude_b, double latitude_b)
    {
        double x = (longitude_b - longitude_a) * Math.cos((latitude_a + latitude_b) / 2);
        double y = (latitude_b - latitude_a);
        return Math.sqrt(Math.pow(x, 2) + Math.pow(y, 2)) * 6371;
    }

    public static double get_radians(String to_parse)
    {
        return Double.parseDouble(to_parse.replace(',', '.')) * Math.PI / 180;
    }

    public static void main(String args[])
    {
        try
        {
            System.setIn(new FileInputStream("input"));
        }
        catch (Exception ignored)
        {
        }

        Scanner in = new Scanner(System.in);
        final double user_longitude = get_radians(in.next());
        final double user_latitude = get_radians(in.next());
        int n = in.nextInt();
        in.nextLine();
        List<Defibrillator> defibrillator_list = new ArrayList<>();
        for (int i = 0; i < n; ++i)
        {
            String[] next_description = in.nextLine().split(";");
            double longitude = get_radians(next_description[4]);
            double latitude = get_radians(next_description[5]);
            Defibrillator defibrillator = new Defibrillator(next_description[1], longitude, latitude);
            defibrillator_list.add(defibrillator);
        }

        System.out.println(defibrillator_list.stream().
                min((a, b) -> Double.compare(distance(user_longitude, user_latitude, a.longitude, a.latitude),
                                distance(user_longitude, user_latitude, b.longitude, b.latitude))).get().name);
    }
}