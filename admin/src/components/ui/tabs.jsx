import * as React from "react"
import { cn } from "@/lib/utils"

const Tabs = ({ defaultValue, value, onValueChange, children, className }) => {
  const [tab, setTab] = React.useState(value ?? defaultValue)
  const active = value ?? tab
  const change = onValueChange ?? setTab
  return (
    <div className={cn("flex flex-col gap-2", className)}>
      {React.Children.map(children, child =>
        React.cloneElement(child, { activeValue: active, onValueChange: change })
      )}
    </div>
  )
}

const TabsList = React.forwardRef(({ className, children, activeValue, onValueChange, ...props }, ref) => (
  <div ref={ref} className={cn("inline-flex h-9 items-center justify-center rounded-lg bg-muted p-1 text-muted-foreground", className)} {...props}>
    {React.Children.map(children, child =>
      React.cloneElement(child, { activeValue, onValueChange })
    )}
  </div>
))
TabsList.displayName = "TabsList"

const TabsTrigger = React.forwardRef(({ className, value, activeValue, onValueChange, children, ...props }, ref) => {
  const active = activeValue === value
  return (
    <button
      ref={ref}
      onClick={() => onValueChange?.(value)}
      className={cn(
        "inline-flex items-center justify-center whitespace-nowrap rounded-md px-3 py-1 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
        active ? "bg-background text-foreground shadow" : "hover:bg-background/50 hover:text-foreground",
        className
      )}
      {...props}
    >
      {children}
    </button>
  )
})
TabsTrigger.displayName = "TabsTrigger"

const TabsContent = React.forwardRef(({ className, value, activeValue, children, ...props }, ref) => {
  if (activeValue !== value) return null
  return (
    <div ref={ref} className={cn("ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2", className)} {...props}>
      {children}
    </div>
  )
})
TabsContent.displayName = "TabsContent"

export { Tabs, TabsList, TabsTrigger, TabsContent }
