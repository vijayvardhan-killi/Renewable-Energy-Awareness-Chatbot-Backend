
export async function postData(question) {
    const baseUrl = import.meta.env.VITE_API || "http://127.0.0.1:8000";
    try {
      const response = await fetch(`${baseUrl}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({"question" : question})
      });
  
      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }
  
      const result = await response.json();
      return result;
  
    } catch (error) {
      throw new Error(error.message || 'Failed to connect to API');
    }
  }
  
// console.log(postData("What is the capital of France?")); // Test the function with a sample question