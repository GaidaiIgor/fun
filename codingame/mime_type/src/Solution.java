import java.io.FileInputStream;
import java.util.*;

class Solution
{
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
        int n = in.nextInt(); // Number of elements which make up the association table.
        int q = in.nextInt(); // Number Q of file names to be analyzed.
        in.nextLine();

        Map<String, String> associations = new HashMap<>();
        for (int i = 0; i < n; ++i)
        {
            String extension = in.next().toLowerCase(); // file extension
            String mime_type = in.next(); // MIME type.
            associations.put(extension, mime_type);
            in.nextLine();
        }

        for (int i = 0; i < q; ++i)
        {
            String file_name = in.nextLine(); // One file name per line.
            int last_dot_index = file_name.lastIndexOf('.');
            if (last_dot_index == -1)
            {
                System.out.println("UNKNOWN");
                continue;
            }
            String extension = file_name.substring(last_dot_index + 1).toLowerCase();
            System.out.println(associations.getOrDefault(extension, "UNKNOWN"));
        }
    }
}