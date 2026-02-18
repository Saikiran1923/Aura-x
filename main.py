from agents.planner import create_plan
from agents.coder import generate_file_code, write_file
from agents.debugger import fix_code
from core.executor import run_python_file
import os

if __name__ == "__main__":
    user_input = input("Enter project request: ")

    # 1️⃣ Create Plan
    plan = create_plan(user_input)

    if not plan:
        print("Planning failed.")
        exit()

    print("\nGenerated Plan:\n")
    print(plan)

    project_name = plan["project_name"]

    # 2️⃣ Process Each Task
    for task in plan["tasks"]:
        for file_name in task["files_to_create"]:
            print(f"\nGenerating {file_name}...")

            # Generate code
            code = generate_file_code(file_name, task["description"])

            # Write file
            write_file(project_name, file_name, code)

            # 3️⃣ Run file
            print(f"\nRunning {file_name}...")
            stdout, stderr = run_python_file(project_name, file_name)

            print("STDOUT:")
            print(stdout)

            print("STDERR:")
            print(stderr)

            # 4️⃣ Auto Debug If Error
            if stderr:
                print("Error detected. Attempting auto-fix...")

                file_path = os.path.join("projects", project_name, file_name)

                with open(file_path, "r", encoding="utf-8") as f:
                    original_code = f.read()

                fixed_code = fix_code(original_code, stderr)

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(fixed_code)

                print("Re-running after fix...")
                stdout, stderr = run_python_file(project_name, file_name)

                print("STDOUT:")
                print(stdout)

                print("STDERR:")
                print(stderr)

                if stderr:
                    print("Still errors after fix.")
                else:
                    print("Fix successful.")
            else:
                print("Execution successful.")

    print("\nProject generation complete.")
