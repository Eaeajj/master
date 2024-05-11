import crypto from "crypto";

export const genId = () => crypto.randomBytes(16).toString("hex");
export const sleep = async (milliseconds: number) => {
  return new Promise((res, rej) => {
    setTimeout(() => {
      res(null);
    }, milliseconds);
  });
};

export const fakeExecution = (timeToExecute: number): Promise<void> => {
  return new Promise((res, rej) => {
    setTimeout(() => {
      res();
    }, timeToExecute);
  });
};

export const numToUpperCaseAscii = (num: number) =>
  String.fromCharCode(65 + num);
