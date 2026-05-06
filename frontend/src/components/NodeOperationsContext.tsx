import { createContext, useContext, type ReactNode } from "react";

export interface NodeOperations {
  updateNodeParams: (nodeId: string, params: Record<string, unknown>) => void;
}

const Context = createContext<NodeOperations | null>(null);

export function NodeOperationsProvider({
  value,
  children,
}: {
  value: NodeOperations;
  children: ReactNode;
}) {
  return <Context.Provider value={value}>{children}</Context.Provider>;
}

/**
 * Returns null when no provider is mounted (e.g. component rendered outside
 * the canvas). Callers should guard before invoking.
 */
export function useNodeOperations(): NodeOperations | null {
  return useContext(Context);
}
