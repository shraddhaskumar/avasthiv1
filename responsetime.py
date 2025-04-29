import sqlite3
import matplotlib.pyplot as plt

# Connect to your SQLite database
conn = sqlite3.connect("users.db")  # Replace with actual database name
cursor = conn.cursor()

# Query all response times
cursor.execute("SELECT response_time FROM chatbot_metrics")
response_times = [row[0] for row in cursor.fetchall()]

# Close the connection
conn.close()

# Check if data exists
if not response_times:
    print("No response time data available.")
else:
    avg_response_time = sum(response_times) / len(response_times)

    # Plot the response times
    plt.figure(figsize=(10, 5))
    plt.hist(response_times, bins=10, color='blue', edgecolor='black', alpha=0.7)
    plt.axvline(avg_response_time, color='red', linestyle='dashed', linewidth=2,
                label=f'Avg: {avg_response_time:.2f} sec')
    plt.xlabel("Response Time (seconds)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Chatbot Response Times")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.show()
