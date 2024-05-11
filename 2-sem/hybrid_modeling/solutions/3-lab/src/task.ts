import { genId } from "./utils";

export type TaskComplexity = "low" | "medium" | "high";
export type TaskPriority = "low" | "medium" | "high";

export type Task = {
  id: string;
  complexity: TaskComplexity;
  priority: TaskPriority;
  createdAt: Date;
  status: "created" | "in-progress" | "completed";

  startedToExecute: Date | null;
  finishedAt: Date | null;
  completedByWorker: string | null;
};

export const createTask = (data?: Partial<Task>) => {
  const baseData: Task = {
    id: genId(),
    complexity: "low",
    priority: "low",
    createdAt: new Date(),
    finishedAt: null,
    startedToExecute: null,
    status: "created",
    completedByWorker: null,
  };
  return Object.assign({}, baseData, { ...data });
};
