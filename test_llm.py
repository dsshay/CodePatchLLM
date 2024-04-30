from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers
import torch

model_id = "codellama/CodeLlama-7b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token 
model = AutoModelForCausalLM.from_pretrained(
   model_id,
   torch_dtype=torch.float16,
   device_map="auto",
)

chat = [
        {"role": "system",
         "content": "You are a helpful and honest code assistant expert in Java. Please, provide all answers to programming questions in Java. Write only code."},
        {"role": "user",
         "content": "Given an array of integers `nums` and an integer `target`, return _indices of the two numbers such that they add up to `target`_. You may assume that each input would have **_exactly_ one solution**, and you may not use the _same_ element twice. You can return the answer in any order. **Example 1:** **Input:** nums = \[2,7,11,15\], target = 9 **Output:** \[0,1\] **Explanation:** Because nums\[0\] + nums\[1\] == 9, we return \[0, 1\]. **Example 2:** **Input:** nums = \[3,2,4\], target = 6 **Output:** \[1,2\] **Example 3:** **Input:** nums = \[3,3\], target = 6 **Output:** \[0,1\] **Constraints:** * `2 <= nums.length <= 104` * `-109 <= nums[i] <= 109` * `-109 <= target <= 109` * **Only one valid answer exists.** **Follow-up:** Can you come up with an algorithm that is less than `O(n2)` time complexity?"},
        {"role": "assistant",
         "content": "public int[] twoSum(int[] nums, int target) {    Map<Integer, Integer> map = new HashMap<>();    for (int i = 0; i < nums.length; i++) {        map.put(nums[i], i);    }    for (int i = 0; i < nums.length; i++) {        int complement = target - nums[i];        if (map.containsKey(complement) && map.get(complement) != i) {            return new int[] {i, map.get(complement)};        }    }    return null;}"},
        {"role": "user",
            "content": "error: cannot find symbol    Map<Integer, Integer> map = new HashMap<>();    ^  symbol:   class Map  location: class Solution, \n error: cannot find symbol    Map<Integer, Integer> map = new HashMap<>();                                    ^  symbol:   class HashMap  location: class Solution"},
        # {"role": "user",
        #  "content": "error: ‘vector’ does not name a type    1 | class Solution{public:vector<int> twoSum(vector<int>& nums, int target) {      |                       ^~~~~~"},
        # {"role": "assistant",
        #  "content": "I apologize for the confusion. It seems that you are using a C++ compiler to compile a C# code. The `vector` type is a part of the C++ Standard Template Library (STL), and it is not available in C#.To fix this error, you can use the `List` class in C# instead of `vector`. Here's the corrected code:```class Solution{    public List<int> twoSum(List<int> nums, int target)    {        var map = new Dictionary<int, int>();        for (int i = 0; i < nums.Count; i++)        {            int complement = target - nums[i];            if (map.ContainsKey(complement))            {                return new List<int> { i, map[complement] };            }            map[nums[i]] = i;        }        return new List<int>();    }}```I hope this helps! Let me know if you have any further questions."},
        # {"role": "user", "content": "I use g++ compiler and I need the c++ code."},
        # {"role": "assistant",
        #  "content": "#include <vector>std::vector<int> twoSum(std::vector<int>& nums, int target) {std::vector<int> result;    for (int i = 0; i < nums.size(); i++) {        int complement = target - nums[i];        if (nums.find(complement) != nums.end()) {            result.push_back(i);            result.push_back(nums.find(complement));            return result;        }    }    return result;}"},
        # {"role": "user", "content": "‘class std::vector<int>’ has no member named ‘find’"},
        # {"role": "assistant",
        #  "content": "#include <vector>#include <algorithm>std::vector<int> twoSum(std::vector<int>& nums, int target) {    std::vector<int> result;    for (int i = 0; i < nums.size(); i++) {        int complement = target - nums[i];        auto it = std::find(nums.begin(), nums.end(), complement);        if (it != nums.end()) {            result.push_back(i);            result.push_back(std::distance(nums.begin(), it));            return result;        }    }    return result;}"},
        # {"role": "user", "content": " use of undeclared identifier 'Solution'"},
        # {"role": "assistant",
        #  "content": "#include <vector>#include <algorithm>class Solution {public:    std::vector<int> twoSum(std::vector<int>& nums, int target) {        std::vector<int> result;        for (int i = 0; i < nums.size(); i++) {            int complement = target - nums[i];            auto it = std::find(nums.begin(), nums.end(), complement);            if (it != nums.end()) {                result.push_back(i);                result.push_back(std::distance(nums.begin(), it));                return result;            }        }        return result;    }};"},
        # {"role": "user", "content": "in function `_start':(.text+0x1b): undefined reference to `main'"},
    ]

inputs = tokenizer.apply_chat_template(chat, return_tensors="pt").to("cuda")

output = model.generate(input_ids=inputs, max_new_tokens=500)
output = output[0].to("cpu")
print(tokenizer.decode(output))

